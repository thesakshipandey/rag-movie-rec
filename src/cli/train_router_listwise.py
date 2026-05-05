#!/usr/bin/env python
"""
Train contextual hedge router with listwise learning (ListMLE loss).

Usage:
    python -m src.cli.train_router_listwise \
        --expert_scores artifacts/router/listwise_expert_scores.parquet \
        --prompts_path projects/Data/prompts.json \
        --out artifacts/router/router_listwise.pt \
        --loss listmle \
        --epochs 50 \
        --lr 1e-4 \
        --batch_size 32
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.router.contextual_hedge_router import ContextualHedgeRouterWithEncoder
from src.router.losses import (
    listmle_loss, listnet_loss, approx_ndcg_loss, 
    compute_expert_weighted_scores
)


class ListwiseDataset(Dataset):
    """Dataset for listwise router training."""
    
    def __init__(
        self,
        expert_scores_df: pd.DataFrame,
        prompts_df: pd.DataFrame,
        prompt_ids: List[str]
    ):
        """
        Args:
            expert_scores_df: DataFrame with expert scores per prompt-movie
            prompts_df: DataFrame with prompt metadata
            prompt_ids: List of prompt IDs to include in this split
        """
        self.expert_scores_df = expert_scores_df
        self.prompts_df = prompts_df
        self.prompt_ids = prompt_ids
        
        # Build vocabularies for categorical features
        self.category_vocab = self._build_vocab(prompts_df, 'category')
        self.difficulty_vocab = self._build_vocab(prompts_df, 'difficulty_mix')
        self.primary_expert_vocab = self._build_vocab(prompts_df, 'primary_expert')
        self.length_bucket_vocab = self._build_vocab(prompts_df, 'length_bucket')
        self.persona_style_vocab = self._build_vocab(prompts_df, 'persona_style')
        
        # Cache prompt data
        self.prompt_data_cache = {}
        for pid in prompt_ids:
            self.prompt_data_cache[pid] = self._prepare_prompt_data(pid)
    
    def _build_vocab(self, df: pd.DataFrame, col: str) -> Dict[str, int]:
        """Build vocabulary for a categorical column."""
        if col not in df.columns:
            return {'UNK': 0}
        unique_vals = df[col].dropna().unique().tolist()
        vocab = {val: idx for idx, val in enumerate(unique_vals)}
        vocab['UNK'] = len(vocab)
        return vocab
    
    def _prepare_prompt_data(self, prompt_id: str) -> Dict:
        """Prepare all data for a single prompt."""
        # Get expert scores for this prompt
        prompt_scores = self.expert_scores_df[
            self.expert_scores_df['prompt_id'] == prompt_id
        ].sort_values('rank')
        
        if len(prompt_scores) == 0:
            return None
        
        # Expert z-scores [N, 4]
        expert_z = np.stack([
            prompt_scores['z_alpha'].values,
            prompt_scores['z_beta'].values,
            prompt_scores['z_gamma'].values,
            prompt_scores['z_delta'].values
        ], axis=1).astype(np.float32)
        
        # Ground truth scores [N]
        gt_scores = prompt_scores['ground_truth_score'].values.astype(np.float32)
        
        # Try to find prompt metadata (by matching prompt_id if possible)
        # Since prompts.json has UUID but merged_all has numeric IDs, we need to handle this
        # For now, use default values
        emotion_dist = np.ones(8, dtype=np.float32) / 8.0  # uniform default
        mix_weights = np.ones(4, dtype=np.float32) / 4.0  # uniform default
        
        # Try to find matching prompt in prompts_df
        # This requires proper ID mapping which depends on your data structure
        # PLACEHOLDER: In production, implement proper mapping
        
        # Extract categorical indices
        category_idx = 0  # default
        difficulty_idx = 0
        primary_expert_idx = 0
        length_bucket_idx = 0
        persona_style_idx = 0
        
        # Extract numerical features
        numerical = np.array([
            10.0,  # length_words (default)
            0.0,   # num_genre_terms
            0.0,   # has_negation
            0.0,   # has_year
            0.0,   # has_actor_or_director
            0.0,   # mentions_specific_movie
            0.0,   # multi_intent
        ], dtype=np.float32)
        
        return {
            'expert_z': expert_z,
            'gt_scores': gt_scores,
            'emotion': emotion_dist,
            'mix_weights': mix_weights,
            'category_idx': category_idx,
            'difficulty_idx': difficulty_idx,
            'primary_expert_idx': primary_expert_idx,
            'length_bucket_idx': length_bucket_idx,
            'persona_style_idx': persona_style_idx,
            'numerical': numerical,
        }
    
    def __len__(self):
        return len(self.prompt_ids)
    
    def __getitem__(self, idx):
        prompt_id = self.prompt_ids[idx]
        data = self.prompt_data_cache.get(prompt_id)
        
        if data is None:
            # Return dummy data (will be filtered)
            return None
        
        return {
            'prompt_id': prompt_id,
            'expert_z': torch.from_numpy(data['expert_z']),
            'gt_scores': torch.from_numpy(data['gt_scores']),
            'features': {
                'emotion': torch.from_numpy(data['emotion']),
                'mix_weights': torch.from_numpy(data['mix_weights']),
                'category_idx': torch.tensor(data['category_idx'], dtype=torch.long),
                'difficulty_idx': torch.tensor(data['difficulty_idx'], dtype=torch.long),
                'primary_expert_idx': torch.tensor(data['primary_expert_idx'], dtype=torch.long),
                'length_bucket_idx': torch.tensor(data['length_bucket_idx'], dtype=torch.long),
                'persona_style_idx': torch.tensor(data['persona_style_idx'], dtype=torch.long),
                'numerical': torch.from_numpy(data['numerical']),
            }
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length lists."""
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Since each prompt has different number of movies, we can't stack
    # Return list of dicts
    return batch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn_name: str,
    device: torch.device,
    entropy_weight: float = 0.0,
    entropy_target: float = 1.2
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_entropy_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None:
            continue
        
        batch_loss = 0.0
        batch_entropy = 0.0
        
        for item in batch:
            # Move to device
            expert_z = item['expert_z'].to(device)  # [N, 4]
            gt_scores = item['gt_scores'].to(device)  # [N]
            
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in item['features'].items()}
            
            # Get router weights
            expert_weights = model(features)  # [4]
            
            # Combine expert scores
            # expert_z: [N, 4], expert_weights: [4]
            # We need to expand expert_weights to [1, 4] and expert_z to [N, 4]
            predicted_scores = (expert_z * expert_weights.unsqueeze(0)).sum(dim=1)  # [N]
            
            # Compute loss
            predicted_scores = predicted_scores.unsqueeze(0)  # [1, N]
            gt_scores = gt_scores.unsqueeze(0)  # [1, N]
            
            if loss_fn_name == "listmle":
                loss = listmle_loss(predicted_scores, gt_scores)
            elif loss_fn_name == "listnet":
                loss = listnet_loss(predicted_scores, gt_scores)
            elif loss_fn_name == "approx_ndcg":
                loss = approx_ndcg_loss(predicted_scores, gt_scores)
            else:
                raise ValueError(f"Unknown loss: {loss_fn_name}")
            
            # Entropy regularization (prevent collapse to single expert)
            if entropy_weight > 0:
                entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum()
                entropy_penalty = torch.relu(entropy_target - entropy)
                loss = loss + entropy_weight * entropy_penalty
                batch_entropy += entropy.item()
            
            batch_loss += loss.item()
        
        # Average over batch
        if len(batch) > 0:
            avg_loss = batch_loss / len(batch)
            avg_entropy = batch_entropy / len(batch) if entropy_weight > 0 else 0.0
            
            # Backward
            optimizer.zero_grad()
            loss = torch.tensor(avg_loss, requires_grad=True)
            
            # Re-compute loss properly with gradients
            batch_total_loss = 0.0
            for item in batch:
                expert_z = item['expert_z'].to(device)
                gt_scores = item['gt_scores'].to(device)
                features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in item['features'].items()}
                
                expert_weights = model(features)
                predicted_scores = (expert_z * expert_weights.unsqueeze(0)).sum(dim=1).unsqueeze(0)
                gt_scores_batch = gt_scores.unsqueeze(0)
                
                if loss_fn_name == "listmle":
                    loss_item = listmle_loss(predicted_scores, gt_scores_batch)
                elif loss_fn_name == "listnet":
                    loss_item = listnet_loss(predicted_scores, gt_scores_batch)
                else:
                    loss_item = approx_ndcg_loss(predicted_scores, gt_scores_batch)
                
                if entropy_weight > 0:
                    entropy = -(expert_weights * torch.log(expert_weights + 1e-8)).sum()
                    entropy_penalty = torch.relu(entropy_target - entropy)
                    loss_item = loss_item + entropy_weight * entropy_penalty
                
                batch_total_loss += loss_item
            
            batch_total_loss = batch_total_loss / len(batch)
            batch_total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += avg_loss
            total_entropy_loss += avg_entropy
            num_batches += 1
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'entropy': total_entropy_loss / max(num_batches, 1)
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn_name: str,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate on validation/test set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    expert_usage = np.zeros(4)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None:
                continue
            
            batch_loss = 0.0
            
            for item in batch:
                expert_z = item['expert_z'].to(device)
                gt_scores = item['gt_scores'].to(device)
                features = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in item['features'].items()}
                
                expert_weights = model(features)
                predicted_scores = (expert_z * expert_weights.unsqueeze(0)).sum(dim=1).unsqueeze(0)
                gt_scores_batch = gt_scores.unsqueeze(0)
                
                if loss_fn_name == "listmle":
                    loss = listmle_loss(predicted_scores, gt_scores_batch)
                elif loss_fn_name == "listnet":
                    loss = listnet_loss(predicted_scores, gt_scores_batch)
                else:
                    loss = approx_ndcg_loss(predicted_scores, gt_scores_batch)
                
                batch_loss += loss.item()
                expert_usage += expert_weights.cpu().numpy()
            
            if len(batch) > 0:
                total_loss += batch_loss / len(batch)
                num_batches += 1
    
    expert_usage = expert_usage / max(num_batches, 1)
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'expert_usage': expert_usage.tolist()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_scores", required=True, help="Parquet with expert scores")
    ap.add_argument("--prompts_path", required=True, help="Path to prompts.json")
    ap.add_argument("--out", default="artifacts/router/router_listwise.pt")
    ap.add_argument("--loss", default="listmle", choices=["listmle", "listnet", "approx_ndcg"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_context", type=int, default=128)
    ap.add_argument("--d_hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--entropy_weight", type=float, default=0.001)
    ap.add_argument("--entropy_target", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    expert_scores_df = pd.read_parquet(args.expert_scores)
    
    with open(args.prompts_path, 'r') as f:
        prompts_list = json.load(f)
    prompts_df = pd.DataFrame(prompts_list)
    
    print(f"Loaded {len(expert_scores_df)} expert score rows")
    print(f"Loaded {len(prompts_df)} prompts")
    
    # Get unique prompt IDs
    all_prompt_ids = expert_scores_df['prompt_id'].unique().tolist()
    print(f"Unique prompts in expert scores: {len(all_prompt_ids)}")
    
    # Train/val/test split (70/15/15) with random shuffle
    print("\n[2/4] Creating train/val/test splits (70/15/15)...")
    np.random.shuffle(all_prompt_ids)
    
    n_train = int(0.70 * len(all_prompt_ids))
    n_val = int(0.15 * len(all_prompt_ids))
    
    train_ids = all_prompt_ids[:n_train]
    val_ids = all_prompt_ids[n_train:n_train+n_val]
    test_ids = all_prompt_ids[n_train+n_val:]
    
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    # Create datasets
    train_dataset = ListwiseDataset(expert_scores_df, prompts_df, train_ids)
    val_dataset = ListwiseDataset(expert_scores_df, prompts_df, val_ids)
    test_dataset = ListwiseDataset(expert_scores_df, prompts_df, test_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    
    # Initialize model
    print("\n[3/4] Initializing model...")
    encoder_kwargs = {
        'include_mix_features': False,
        'category_vocab': train_dataset.category_vocab.keys(),
        'difficulty_vocab': train_dataset.difficulty_vocab.keys(),
        'primary_expert_vocab': train_dataset.primary_expert_vocab.keys(),
        'length_bucket_vocab': train_dataset.length_bucket_vocab.keys(),
        'persona_style_vocab': train_dataset.persona_style_vocab.keys(),
    }
    
    model = ContextualHedgeRouterWithEncoder(
        d_context=args.d_context,
        d_hidden=args.d_hidden,
        num_experts=4,
        dropout=args.dropout,
        temperature=args.temperature,
        encoder_kwargs=encoder_kwargs
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n[4/4] Training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.loss, device,
            entropy_weight=args.entropy_weight,
            entropy_target=args.entropy_target
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, args.loss, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | Entropy: {train_metrics['entropy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Expert Usage (val): α={val_metrics['expert_usage'][0]:.3f}, "
              f"β={val_metrics['expert_usage'][1]:.3f}, "
              f"γ={val_metrics['expert_usage'][2]:.3f}, "
              f"δ={val_metrics['expert_usage'][3]:.3f}")
        print(f"Temperature: {model.get_temperature():.3f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            
            # Save model with vocabularies for inference
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'expert_usage': val_metrics['expert_usage'],
                'args': vars(args),
                'vocabularies': {
                    'category': train_dataset.category_vocab,
                    'difficulty': train_dataset.difficulty_vocab,
                    'primary_expert': train_dataset.primary_expert_vocab,
                    'length_bucket': train_dataset.length_bucket_vocab,
                    'persona_style': train_dataset.persona_style_vocab,
                }
            }, args.out)
            
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best epoch: {best_epoch} | Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.out}")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(args.out)['model_state_dict'])
    test_metrics = evaluate(model, test_loader, args.loss, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Expert Usage (test): α={test_metrics['expert_usage'][0]:.3f}, "
          f"β={test_metrics['expert_usage'][1]:.3f}, "
          f"γ={test_metrics['expert_usage'][2]:.3f}, "
          f"δ={test_metrics['expert_usage'][3]:.3f}")


if __name__ == "__main__":
    main()

