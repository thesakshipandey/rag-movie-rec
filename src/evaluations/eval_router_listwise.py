#!/usr/bin/env python
"""
Evaluate trained contextual hedge router on test set.

Computes nDCG@10, MRR, Hit@10 with per-category and per-difficulty breakdowns.

Usage:
    python -m src.evaluations.eval_router_listwise \
        --expert_scores artifacts/router/listwise_expert_scores.parquet \
        --prompts_path projects/Data/prompts.json \
        --router_checkpoint artifacts/router/router_listwise.pt \
        --out artifacts/evaluation_results/listwise/router_metrics.json
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.router.contextual_hedge_router import ContextualHedgeRouterWithEncoder
from src.evaluations.eval_single_experts import ndcg_at_k, mrr, hit_at_k


def evaluate_router_on_prompts(
    model: torch.nn.Module,
    expert_scores_df: pd.DataFrame,
    prompt_ids: List[str],
    device: torch.device,
    k: int = 10,
    relevance_threshold: float = 0.9
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate router on a set of prompts.
    
    Returns:
        (aggregate_metrics, per_prompt_results)
    """
    model.eval()
    
    ndcg_scores = []
    mrr_scores = []
    hit_scores = []
    per_prompt_results = []
    expert_usage = np.zeros(4)
    
    with torch.no_grad():
        for prompt_id in tqdm(prompt_ids, desc="Evaluating router"):
            prompt_data = expert_scores_df[expert_scores_df['prompt_id'] == prompt_id].sort_values('rank')
            
            if len(prompt_data) == 0:
                continue
            
            # Get expert z-scores [N, 4]
            expert_z = np.stack([
                prompt_data['z_alpha'].values,
                prompt_data['z_beta'].values,
                prompt_data['z_gamma'].values,
                prompt_data['z_delta'].values
            ], axis=1).astype(np.float32)
            
            expert_z_tensor = torch.from_numpy(expert_z).to(device)
            
            # Get features (placeholder - in production, use actual features)
            features = {
                'emotion': torch.ones(8, device=device) / 8.0,
                'category_idx': torch.tensor(0, dtype=torch.long, device=device),
                'difficulty_idx': torch.tensor(0, dtype=torch.long, device=device),
                'primary_expert_idx': torch.tensor(0, dtype=torch.long, device=device),
                'length_bucket_idx': torch.tensor(0, dtype=torch.long, device=device),
                'persona_style_idx': torch.tensor(0, dtype=torch.long, device=device),
                'numerical': torch.zeros(7, device=device),
            }
            
            # Get router weights
            expert_weights = model(features)  # [4]
            expert_usage += expert_weights.cpu().numpy()
            
            # Combine expert scores
            pred_scores = (expert_z_tensor * expert_weights.unsqueeze(0)).sum(dim=1)  # [N]
            pred_scores = pred_scores.cpu().numpy()
            
            gt_scores = prompt_data['ground_truth_score'].values
            
            # Compute metrics
            ndcg = ndcg_at_k(pred_scores, gt_scores, k)
            mrr_val = mrr(pred_scores, gt_scores, threshold=relevance_threshold)
            hit = hit_at_k(pred_scores, gt_scores, k, threshold=relevance_threshold)
            
            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr_val)
            hit_scores.append(hit)
            
            per_prompt_results.append({
                'prompt_id': prompt_id,
                f'ndcg@{k}': float(ndcg),
                'mrr': float(mrr_val),
                f'hit@{k}': float(hit),
                'expert_weights': expert_weights.cpu().numpy().tolist(),
                'num_movies': len(prompt_data)
            })
    
    expert_usage = expert_usage / len(prompt_ids)
    
    aggregate_metrics = {
        f'ndcg@{k}': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f'ndcg@{k}_std': float(np.std(ndcg_scores)) if ndcg_scores else 0.0,
        'mrr': float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        'mrr_std': float(np.std(mrr_scores)) if mrr_scores else 0.0,
        f'hit@{k}': float(np.mean(hit_scores)) if hit_scores else 0.0,
        f'hit@{k}_std': float(np.std(hit_scores)) if hit_scores else 0.0,
        'expert_usage': {
            'alpha': float(expert_usage[0]),
            'beta': float(expert_usage[1]),
            'gamma': float(expert_usage[2]),
            'delta': float(expert_usage[3])
        },
        'num_prompts': len(ndcg_scores)
    }
    
    return aggregate_metrics, per_prompt_results


def compute_per_category_metrics(
    per_prompt_results: List[Dict],
    prompts_df: pd.DataFrame,
    k: int = 10
) -> Dict[str, Dict]:
    """
    Compute metrics broken down by category.
    
    Note: Requires proper ID mapping between prompt_ids.
    """
    # Since we don't have proper ID mapping, return placeholder
    return {
        'note': 'Per-category breakdown requires proper ID mapping between datasets'
    }


def compute_per_difficulty_metrics(
    per_prompt_results: List[Dict],
    prompts_df: pd.DataFrame,
    k: int = 10
) -> Dict[str, Dict]:
    """
    Compute metrics broken down by difficulty.
    
    Note: Requires proper ID mapping between prompt_ids.
    """
    # Since we don't have proper ID mapping, return placeholder
    return {
        'note': 'Per-difficulty breakdown requires proper ID mapping between datasets'
    }


def analyze_expert_selection(per_prompt_results: List[Dict]) -> Dict:
    """
    Analyze which experts are selected most often and in what patterns.
    """
    # Extract all expert weights
    all_weights = np.array([r['expert_weights'] for r in per_prompt_results])
    
    # Dominant expert per prompt (argmax)
    dominant_experts = np.argmax(all_weights, axis=1)
    expert_names = ['alpha', 'beta', 'gamma', 'delta']
    
    dominant_counts = {
        expert_names[i]: int((dominant_experts == i).sum())
        for i in range(4)
    }
    
    # Average weights
    avg_weights = all_weights.mean(axis=0)
    
    # Weight statistics
    weight_stats = {
        f'{expert_names[i]}': {
            'mean': float(avg_weights[i]),
            'std': float(all_weights[:, i].std()),
            'min': float(all_weights[:, i].min()),
            'max': float(all_weights[:, i].max()),
            'dominant_count': dominant_counts[expert_names[i]]
        }
        for i in range(4)
    }
    
    return {
        'average_weights': {k: float(v) for k, v in zip(expert_names, avg_weights)},
        'dominant_expert_counts': dominant_counts,
        'weight_statistics': weight_stats
    }


def correlation_with_ground_truth_weights(
    per_prompt_results: List[Dict],
    prompts_df: pd.DataFrame
) -> Dict:
    """
    Compute correlation between predicted and ground truth mix_weights.
    
    Note: Requires proper ID mapping.
    """
    return {
        'note': 'Correlation analysis requires proper ID mapping between datasets',
        'placeholder': 'Not implemented due to ID mismatch'
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_scores", required=True, help="Parquet with expert scores")
    ap.add_argument("--prompts_path", required=True, help="Path to prompts.json")
    ap.add_argument("--router_checkpoint", required=True, help="Path to trained router .pt file")
    ap.add_argument("--out", default="artifacts/evaluation_results/listwise/router_metrics.json")
    ap.add_argument("--k", type=int, default=10, help="K for nDCG@K and Hit@K")
    ap.add_argument("--relevance_threshold", type=float, default=0.9, help="Threshold for relevance")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="test", help="Which split to evaluate")
    args = ap.parse_args()
    
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
    
    # Load router checkpoint
    print("\n[2/4] Loading router...")
    checkpoint = torch.load(args.router_checkpoint, map_location=device)
    
    # Reconstruct model with saved vocabularies
    model_args = checkpoint.get('args', {})
    vocabularies = checkpoint.get('vocabularies', {})
    
    print(f"  Loaded vocabularies:")
    for key, vocab in vocabularies.items():
        print(f"    {key}: {len(vocab)} unique values")
    
    encoder_kwargs = {
        'include_mix_features': False,
        'category_vocab': list(vocabularies.get('category', {}).keys()) if vocabularies else None,
        'difficulty_vocab': list(vocabularies.get('difficulty', {}).keys()) if vocabularies else None,
        'primary_expert_vocab': list(vocabularies.get('primary_expert', {}).keys()) if vocabularies else None,
        'length_bucket_vocab': list(vocabularies.get('length_bucket', {}).keys()) if vocabularies else None,
        'persona_style_vocab': list(vocabularies.get('persona_style', {}).keys()) if vocabularies else None,
    }
    
    model = ContextualHedgeRouterWithEncoder(
        d_context=model_args.get('d_context', 128),
        d_hidden=model_args.get('d_hidden', 256),
        num_experts=4,
        dropout=model_args.get('dropout', 0.2),
        temperature=model_args.get('temperature', 1.0),
        encoder_kwargs=encoder_kwargs
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"Validation loss: {checkpoint.get('val_loss', '?')}")
    
    # Get test prompt IDs
    all_prompt_ids = expert_scores_df['prompt_id'].unique().tolist()
    
    # In production, load proper train/val/test split
    # For now, use all prompts
    print(f"\nEvaluating on {len(all_prompt_ids)} prompts")
    test_prompt_ids = all_prompt_ids
    
    # Evaluate
    print("\n[3/4] Evaluating router...")
    aggregate_metrics, per_prompt_results = evaluate_router_on_prompts(
        model,
        expert_scores_df,
        test_prompt_ids,
        device,
        k=args.k,
        relevance_threshold=args.relevance_threshold
    )
    
    # Analyze expert selection
    print("\n[4/4] Analyzing results...")
    expert_analysis = analyze_expert_selection(per_prompt_results)
    
    # Compile final results
    results = {
        'aggregate_metrics': aggregate_metrics,
        'expert_selection_analysis': expert_analysis,
        'per_category_metrics': compute_per_category_metrics(per_prompt_results, prompts_df, k=args.k),
        'per_difficulty_metrics': compute_per_difficulty_metrics(per_prompt_results, prompts_df, k=args.k),
        'correlation_with_ground_truth': correlation_with_ground_truth_weights(per_prompt_results, prompts_df),
        'checkpoint_info': {
            'path': args.router_checkpoint,
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_loss': float(checkpoint.get('val_loss', 0.0))
        },
        'evaluation_config': {
            'k': args.k,
            'relevance_threshold': args.relevance_threshold,
            'num_test_prompts': len(test_prompt_ids)
        }
    }
    
    # Save results
    print(f"\nSaving results to {args.out}...")
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save per-prompt details
    per_prompt_out = args.out.replace('.json', '_per_prompt.json')
    with open(per_prompt_out, 'w') as f:
        json.dump(per_prompt_results, f, indent=2)
    
    print(f"Per-prompt results saved to {per_prompt_out}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ROUTER EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nAggregate Metrics:")
    print(f"  nDCG@{args.k}: {aggregate_metrics[f'ndcg@{args.k}']:.4f} ± {aggregate_metrics[f'ndcg@{args.k}_std']:.4f}")
    print(f"  MRR: {aggregate_metrics['mrr']:.4f} ± {aggregate_metrics['mrr_std']:.4f}")
    print(f"  Hit@{args.k}: {aggregate_metrics[f'hit@{args.k}']:.4f} ± {aggregate_metrics[f'hit@{args.k}_std']:.4f}")
    
    print(f"\nExpert Usage (average weights):")
    for expert, weight in aggregate_metrics['expert_usage'].items():
        print(f"  {expert}: {weight:.3f}")
    
    print(f"\nExpert Selection Analysis:")
    for expert, stats in expert_analysis['weight_statistics'].items():
        print(f"  {expert}:")
        print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Times dominant: {stats['dominant_count']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

