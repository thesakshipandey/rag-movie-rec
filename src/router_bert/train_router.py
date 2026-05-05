"""
Training script for BERT-based expert router.

Trains a text-conditioned router using Bradley-Terry pairwise loss.
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

# Set HuggingFace cache paths
os.environ['HF_HOME'] = '/mnt/nas/sakshipandey/main/models'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nas/sakshipandey/main/models/transformers'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nas/sakshipandey/main/models/datasets'

from src.router_bert.data.loader import RouterDataset, collate_prompts_fn
from src.router_bert.models.four_head_router import FourHeadRouter
from src.router_bert.utils.metrics import compute_overall_metrics


def bradley_terry_loss(
    weights: torch.Tensor,
    dz_batch: torch.Tensor,
    y_batch: torch.Tensor,
    pair_to_prompt_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Bradley-Terry loss for a batch.
    
    Args:
        weights: [B, 4] expert weights for B prompts
        dz_batch: [N, 4] delta features for N pairs
        y_batch: [N] labels in {0, 1}
        pair_to_prompt_idx: [N] mapping from pair to prompt index
        temperature: Temperature for scaling
    
    Returns:
        loss: scalar loss
    """
    # Get weights for each pair
    pair_weights = weights[pair_to_prompt_idx]  # [N, 4]
    
    # Compute scores: s = w · dz
    scores = torch.sum(pair_weights * dz_batch, dim=-1)  # [N]
    
    # Convert y from {0, 1} to {-1, +1}
    y_prime = 2 * y_batch - 1  # [N]
    
    # Bradley-Terry loss: log(1 + exp(-y' * s / T))
    loss = torch.log(1 + torch.exp(-y_prime * scores / temperature))
    
    return loss.mean()


def entropy_penalty(
    weights: torch.Tensor,
    entropy_min: float = 0.6,
) -> torch.Tensor:
    """
    Compute entropy penalty to encourage diversity.
    
    Args:
        weights: [B, 4] expert weights
        entropy_min: Minimum desired entropy
    
    Returns:
        penalty: scalar penalty
    """
    # H(w) = -sum(w * log(w))
    log_weights = torch.log(weights + 1e-10)
    entropy = -torch.sum(weights * log_weights, dim=-1)  # [B]
    
    # Penalty when entropy is below minimum
    penalty = torch.clamp(entropy_min - entropy, min=0.0)
    
    return penalty.mean()


def train_epoch(
    model: FourHeadRouter,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float,
    entropy_min: float,
    entropy_lambda: float,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_bt_loss = 0.0
    total_entropy_loss = 0.0
    n_batches = 0
    total_pairs = 0
    total_nonzero_scores = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Get prompts and pairs
        prompt_texts = batch['prompt_texts']
        all_pairs = batch['all_pairs']
        pair_to_prompt_idx = batch['pair_to_prompt_idx'].to(device)
        
        # Encode prompts to get weights [B, 4]
        weights, _ = model.forward_texts(prompt_texts, return_attn=False)
        
        # Prepare pair data
        dz_list = [pair['dz'] for pair in all_pairs]
        y_list = [pair['y'] for pair in all_pairs]
        
        dz_batch = torch.tensor(np.stack(dz_list), dtype=torch.float32, device=device)
        y_batch = torch.tensor(y_list, dtype=torch.float32, device=device)
        
        # Compute scores for diagnostics
        pair_weights = weights[pair_to_prompt_idx]
        scores = torch.sum(pair_weights * dz_batch, dim=-1)
        total_pairs += len(scores)
        total_nonzero_scores += (scores.abs() > 1e-6).sum().item()
        
        # Compute Bradley-Terry loss
        bt_loss = bradley_terry_loss(
            weights, dz_batch, y_batch, pair_to_prompt_idx, temperature
        )
        
        # Compute entropy penalty
        ent_penalty = entropy_penalty(weights, entropy_min)
        
        # Total loss
        loss = bt_loss + entropy_lambda * ent_penalty
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track
        total_loss += loss.item()
        total_bt_loss += bt_loss.item()
        total_entropy_loss += ent_penalty.item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bt': f'{bt_loss.item():.4f}',
            'ent': f'{ent_penalty.item():.4f}',
        })
    
    return {
        'loss': total_loss / n_batches,
        'bt_loss': total_bt_loss / n_batches,
        'entropy_loss': total_entropy_loss / n_batches,
        'nonzero_score_ratio': total_nonzero_scores / total_pairs if total_pairs > 0 else 0.0,
    }


@torch.no_grad()
def evaluate(
    model: FourHeadRouter,
    dataloader: DataLoader,
    device: str,
    temperature: float,
    tol: float = 0.05,
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_scores = []
    all_y_prime = []
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        # Get prompts and pairs
        prompt_texts = batch['prompt_texts']
        all_pairs = batch['all_pairs']
        pair_to_prompt_idx = batch['pair_to_prompt_idx'].to(device)
        
        # Encode prompts
        weights, _ = model.forward_texts(prompt_texts, return_attn=False)
        
        # Prepare pair data
        dz_list = [pair['dz'] for pair in all_pairs]
        y_list = [pair['y'] for pair in all_pairs]
        
        dz_batch = torch.tensor(np.stack(dz_list), dtype=torch.float32, device=device)
        y_batch = torch.tensor(y_list, dtype=torch.float32, device=device)
        
        # Compute loss
        bt_loss = bradley_terry_loss(
            weights, dz_batch, y_batch, pair_to_prompt_idx, temperature
        )
        total_loss += bt_loss.item()
        n_batches += 1
        
        # Compute scores for metrics
        pair_weights = weights[pair_to_prompt_idx]
        scores = torch.sum(pair_weights * dz_batch, dim=-1)
        y_prime = 2 * y_batch - 1
        
        all_scores.extend(scores.cpu().numpy())
        all_y_prime.extend(y_prime.cpu().numpy())
    
    # Compute metrics
    all_scores = np.array(all_scores)
    all_y_prime = np.array(all_y_prime)
    
    metrics = compute_overall_metrics(all_scores, all_y_prime, tol=tol)
    metrics['loss'] = total_loss / n_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train BERT-based expert router")
    
    # Data
    parser.add_argument(
        '--parquet',
        type=str,
        default='artifacts/router/features_sum.with_splits.bal.parquet',
        help='Path to features parquet'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        default='artifacts/prompts/prompt_text.parquet',
        help='Path to prompts parquet'
    )
    
    # Model
    parser.add_argument(
        '--encoder',
        type=str,
        default='distilbert-base-uncased',
        help='HuggingFace encoder name'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        default=True,
        help='Freeze encoder (default: True)'
    )
    parser.add_argument(
        '--unfreeze',
        action='store_true',
        help='Fine-tune encoder (overrides --freeze_encoder)'
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=256,
        help='Max sequence length'
    )
    
    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Global learning rate (overrides --head_lr / --encoder_lr when provided)'
    )
    parser.add_argument(
        '--head_lr',
        type=float,
        default=5e-4,
        help='Learning rate for trainable router head parameters'
    )
    parser.add_argument(
        '--encoder_lr',
        type=float,
        default=2e-5,
        help='Learning rate for encoder parameters when unfrozen'
    )
    parser.add_argument(
        '--batch_prompts',
        type=int,
        default=16,
        help='Batch size (number of prompts)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for Bradley-Terry loss'
    )
    parser.add_argument(
        '--entropy_min',
        type=float,
        default=0.6,
        help='Minimum entropy for regularization'
    )
    parser.add_argument(
        '--entropy_lambda',
        type=float,
        default=1e-3,
        help='Weight for entropy penalty'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay for optimizer'
    )
    
    # Output
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Output directory (default: artifacts/router/bert_router/run_<timestamp>)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Handle freeze/unfreeze
    if args.unfreeze:
        args.freeze_encoder = False
    
    # Set output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_dir = f'artifacts/router/bert_router/run_{timestamp}'
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output directory: {args.out_dir}")
    
    # Print configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Encoder: {args.encoder}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print(f"Epochs: {args.epochs}")
    effective_lr = args.lr if args.lr is not None else None
    print(f"Learning rate (global): {effective_lr if effective_lr is not None else 'n/a'}")
    print(f"Head learning rate: {args.head_lr}")
    print(f"Encoder learning rate: {args.encoder_lr}")
    print(f"Batch size (prompts): {args.batch_prompts}")
    print(f"Temperature: {args.temperature}")
    print(f"Entropy min: {args.entropy_min}, lambda: {args.entropy_lambda}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = RouterDataset(args.parquet, args.prompts, split='train')
    val_dataset = RouterDataset(args.parquet, args.prompts, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_prompts,
        shuffle=True,
        collate_fn=collate_prompts_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_prompts,
        shuffle=False,
        collate_fn=collate_prompts_fn,
    )
    
    # Create model
    print("\nInitializing model...")
    model = FourHeadRouter(
        encoder_name=args.encoder,
        freeze_encoder=args.freeze_encoder,
        max_length=args.max_len,
    )
    model.to(args.device)
    
    # Optimizer with per-group learning rates
    if args.lr is not None:
        head_lr = args.lr
        encoder_lr = args.lr
    else:
        head_lr = args.head_lr
        encoder_lr = args.encoder_lr

    head_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
            "weight_decay": args.weight_decay,
        })
    if encoder_params:
        lr_enc = encoder_lr if args.lr is None else args.lr
        param_groups.append({
            "params": encoder_params,
            "lr": lr_enc,
            "weight_decay": args.weight_decay,
        })

    optimizer = torch.optim.AdamW(param_groups)

    actual_head_lr = head_lr if head_params else None
    actual_encoder_lr = (encoder_lr if args.lr is None else args.lr) if encoder_params else None

    print(f"Trainable head params: {sum(p.numel() for p in head_params)} | lr={actual_head_lr}")
    print(f"Trainable encoder params: {sum(p.numel() for p in encoder_params)} | lr={actual_encoder_lr}")

    # Persist configuration (with resolved lrs)
    config = vars(args).copy()
    config.update({
        "head_lr_used": actual_head_lr,
        "encoder_lr_used": actual_encoder_lr,
    })
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    print("\nStarting training...\n")
    best_val_agree = 0.0
    training_log = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            args.temperature, args.entropy_min, args.entropy_lambda,
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, args.device, args.temperature,
        )
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_bt_loss': train_metrics['bt_loss'],
            'train_entropy_loss': train_metrics['entropy_loss'],
            'val_loss': val_metrics['loss'],
            'val_agree_no_ties': val_metrics['agree_no_ties'],
            'val_agree_ties_0p5': val_metrics['agree_ties_0p5'],
        }
        training_log.append(log_entry)
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} "
              f"(BT: {train_metrics['bt_loss']:.4f}, Ent: {train_metrics['entropy_loss']:.4f})")
        print(f"Train nonzero scores: {train_metrics['nonzero_score_ratio']:.2%}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Agree (no ties): {val_metrics['agree_no_ties']:.4f}")
        print(f"Val Agree (ties 0.5): {val_metrics['agree_ties_0p5']:.4f}")
        
        # Save best model
        if val_metrics['agree_no_ties'] > best_val_agree:
            best_val_agree = val_metrics['agree_no_ties']
            print(f"New best validation agreement: {best_val_agree:.4f}")
            model.save_pretrained(os.path.join(args.out_dir, 'best_model'))
        
        # Save latest model
        model.save_pretrained(os.path.join(args.out_dir, 'latest_model'))
    
    # Save training log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(os.path.join(args.out_dir, 'training_log.csv'), index=False)
    print(f"\nTraining log saved to {args.out_dir}/training_log.csv")
    
    print(f"\nTraining complete! Best validation agreement: {best_val_agree:.4f}")
    print(f"Models saved to {args.out_dir}")


if __name__ == '__main__':
    main()
