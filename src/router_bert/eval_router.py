"""
Evaluation script for BERT-based expert router.

Loads a trained model and computes comprehensive metrics and visualizations.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
import random

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set HuggingFace cache paths
os.environ['HF_HOME'] = '/mnt/nas/sakshipandey/main/models'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/nas/sakshipandey/main/models/transformers'
os.environ['HF_DATASETS_CACHE'] = '/mnt/nas/sakshipandey/main/models/datasets'

from src.router_bert.data.loader import load_router_data
from src.router_bert.models.four_head_router import FourHeadRouter
from src.router_bert.utils.metrics import (
    compute_overall_metrics,
    compute_metrics_by_group,
)
from src.router_bert.utils.viz import (
    plot_weights_histogram,
    save_attention_examples,
)


@torch.no_grad()
def evaluate_model(
    model: FourHeadRouter,
    df: pd.DataFrame,
    device: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Evaluate model on a dataset.
    
    Returns DataFrame with predictions added.
    """
    model.eval()
    
    # Group by prompt
    results = []
    
    prompt_groups = list(df.groupby('prompt_id'))
    
    pbar = tqdm(prompt_groups, desc="Evaluating")
    for prompt_id, group in pbar:
        prompt_text = group['text'].iloc[0]
        
        # Get weights for this prompt
        weights, _ = model.forward_texts([prompt_text], return_attn=False)
        weights = weights[0].cpu().numpy()  # [4]
        
        # Compute scores for all pairs
        for _, row in group.iterrows():
            dz = np.array([
                row['dz_alpha'],
                row['dz_beta'],
                row['dz_gamma'],
                row['dz_delta'],
            ])
            
            score = np.dot(weights, dz)
            y_prime = 2 * row['y'] - 1
            
            result = {
                'prompt_id': prompt_id,
                'pair_id': row['pair_id'],
                'y': row['y'],
                'y_prime': y_prime,
                'score': score,
                'weight_alpha': weights[0],
                'weight_beta': weights[1],
                'weight_gamma': weights[2],
                'weight_delta': weights[3],
            }
            
            # Add metadata if present
            if 'difficulty' in row:
                result['difficulty'] = row['difficulty']
            if 'category' in row:
                result['category'] = row['category']
            
            results.append(result)
    
    return pd.DataFrame(results)


@torch.no_grad()
def collect_attention_examples(
    model: FourHeadRouter,
    df: pd.DataFrame,
    device: str,
    n_examples: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Collect attention examples for visualization.
    """
    model.eval()
    
    # Sample random prompts
    unique_prompts = df.groupby('prompt_id')['text'].first()
    random.seed(seed)
    sampled_prompt_ids = random.sample(list(unique_prompts.index), min(n_examples, len(unique_prompts)))
    
    examples = []
    
    for prompt_id in sampled_prompt_ids:
        prompt_text = unique_prompts[prompt_id]
        
        # Tokenize to get input_ids
        encoded = model.tokenizer(
            [prompt_text],
            padding=True,
            truncation=True,
            max_length=model.max_length,
            return_tensors='pt',
        )
        input_ids = encoded['input_ids'][0]
        
        # Get weights and attention
        weights, attn = model.forward_texts(
            [prompt_text],
            return_attn=True,
        )
        
        examples.append({
            'prompt_id': prompt_id,
            'prompt_text': prompt_text,
            'weights': weights[0].cpu().numpy(),
            'attn_weights': attn[0].cpu().numpy(),  # [4, T]
            'input_ids': input_ids.cpu(),  # [T]
            'tokenizer': model.tokenizer,
        })
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate BERT-based expert router")
    
    # Model
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        required=True,
        help='Path to saved model directory'
    )
    
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
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Split to evaluate (train/val/test)'
    )
    
    # Evaluation
    parser.add_argument(
        '--tol',
        type=float,
        default=0.05,
        help='Tolerance for tie detection'
    )
    parser.add_argument(
        '--n_attn_examples',
        type=int,
        default=10,
        help='Number of attention examples to save'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("EVALUATION CONFIGURATION")
    print("="*80)
    print(f"Model checkpoint: {args.ckpt_dir}")
    print(f"Split: {args.split}")
    print(f"Tolerance: {args.tol}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")
    
    # Create output directory
    out_dir = os.path.join(args.ckpt_dir, f'eval_{args.split}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")
    
    # Load model
    print("Loading model...")
    model = FourHeadRouter.from_pretrained(args.ckpt_dir, device=args.device)
    print()
    
    # Load data
    print("Loading data...")
    df = load_router_data(args.parquet, args.prompts, split=args.split)
    print()
    
    # Evaluate
    print("Running evaluation...")
    results_df = evaluate_model(model, df, args.device)
    
    # Save predictions
    results_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    print(f"Saved predictions to {out_dir}/predictions.csv")
    
    # Compute overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    
    overall_metrics = compute_overall_metrics(
        results_df['score'].values,
        results_df['y_prime'].values,
        tol=args.tol,
    )
    
    for key, value in overall_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save overall metrics
    with open(os.path.join(out_dir, 'metrics_overall.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    # Compute metrics by difficulty
    if 'difficulty' in results_df.columns:
        print("\n" + "="*80)
        print("METRICS BY DIFFICULTY")
        print("="*80)
        
        difficulty_metrics = compute_metrics_by_group(
            results_df,
            'difficulty',
            tol=args.tol,
        )
        print(difficulty_metrics.to_string(index=False))
        
        difficulty_metrics.to_csv(
            os.path.join(out_dir, 'metrics_by_difficulty.csv'),
            index=False,
        )
    
    # Compute metrics by category
    if 'category' in results_df.columns:
        print("\n" + "="*80)
        print("METRICS BY CATEGORY")
        print("="*80)
        
        category_metrics = compute_metrics_by_group(
            results_df,
            'category',
            tol=args.tol,
        )
        print(category_metrics.to_string(index=False))
        
        category_metrics.to_csv(
            os.path.join(out_dir, 'metrics_by_category.csv'),
            index=False,
        )
    
    # Plot weight distributions
    print("\n" + "="*80)
    print("WEIGHT DISTRIBUTIONS")
    print("="*80)
    
    weights_array = results_df[['weight_alpha', 'weight_beta', 'weight_gamma', 'weight_delta']].values
    
    # Get unique weights (one per prompt)
    unique_weights = results_df.groupby('prompt_id')[
        ['weight_alpha', 'weight_beta', 'weight_gamma', 'weight_delta']
    ].first().values
    
    print(f"Alpha  - Mean: {unique_weights[:, 0].mean():.4f}, Std: {unique_weights[:, 0].std():.4f}")
    print(f"Beta   - Mean: {unique_weights[:, 1].mean():.4f}, Std: {unique_weights[:, 1].std():.4f}")
    print(f"Gamma  - Mean: {unique_weights[:, 2].mean():.4f}, Std: {unique_weights[:, 2].std():.4f}")
    print(f"Delta  - Mean: {unique_weights[:, 3].mean():.4f}, Std: {unique_weights[:, 3].std():.4f}")
    
    plot_weights_histogram(
        unique_weights,
        os.path.join(out_dir, 'weights_histogram.png'),
    )
    
    # Collect and save attention examples
    print("\n" + "="*80)
    print("ATTENTION EXAMPLES")
    print("="*80)
    
    examples = collect_attention_examples(
        model, df, args.device, n_examples=args.n_attn_examples,
    )
    
    attn_dir = os.path.join(out_dir, 'attn_examples')
    save_attention_examples(examples, attn_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"All results saved to: {out_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

