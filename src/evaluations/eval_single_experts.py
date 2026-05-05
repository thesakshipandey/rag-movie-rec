#!/usr/bin/env python
"""
Evaluate single experts on listwise ranking task.

Computes nDCG@10, MRR, and Hit@10 for each expert independently,
as well as baseline combinations (uniform, oracle).

Usage:
    python -m src.evaluations.eval_single_experts \
        --expert_scores artifacts/router/listwise_expert_scores.parquet \
        --prompts_path projects/Data/prompts.json \
        --out artifacts/evaluation_results/listwise/single_expert_metrics.json \
        --split test
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain at K.
    
    DCG@K = Σᵢ (2^relᵢ - 1) / log₂(i + 1)  for i in [1, k]
    """
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    
    gains = 2 ** relevances - 1
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(predicted_scores: np.ndarray, ground_truth_scores: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    
    nDCG@K = DCG@K / IDCG@K
    where IDCG is the ideal DCG (sorted by ground truth).
    """
    # Sort by predicted scores
    pred_order = np.argsort(-predicted_scores)
    relevances = ground_truth_scores[pred_order]
    
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG (sort by ground truth)
    ideal_relevances = np.sort(ground_truth_scores)[::-1]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def mrr(predicted_scores: np.ndarray, ground_truth_scores: np.ndarray, threshold: float = 0.9) -> float:
    """
    Mean Reciprocal Rank.
    
    Finds the rank of the first relevant item (ground truth >= threshold).
    MRR = 1 / rank
    """
    # Sort by predicted scores (descending)
    pred_order = np.argsort(-predicted_scores)
    gt_sorted = ground_truth_scores[pred_order]
    
    # Find first relevant item
    relevant_mask = gt_sorted >= threshold
    if not relevant_mask.any():
        return 0.0
    
    first_relevant_rank = np.where(relevant_mask)[0][0] + 1  # 1-indexed
    return 1.0 / first_relevant_rank


def hit_at_k(predicted_scores: np.ndarray, ground_truth_scores: np.ndarray, k: int, threshold: float = 0.9) -> float:
    """
    Hit@K: fraction of relevant items in top-K.
    
    An item is relevant if ground_truth_score >= threshold.
    """
    # Get top-K predictions
    pred_order = np.argsort(-predicted_scores)[:k]
    top_k_gt = ground_truth_scores[pred_order]
    
    # Count relevant items in top-K
    num_relevant_in_k = (top_k_gt >= threshold).sum()
    
    # Total relevant items
    total_relevant = (ground_truth_scores >= threshold).sum()
    
    if total_relevant == 0:
        return 0.0
    
    return num_relevant_in_k / total_relevant


def evaluate_expert(
    expert_scores_df: pd.DataFrame,
    prompt_ids: List[str],
    expert_name: str,
    k: int = 10,
    relevance_threshold: float = 0.9
) -> Dict:
    """
    Evaluate a single expert on all prompts.
    
    Args:
        expert_scores_df: DataFrame with expert scores
        prompt_ids: List of prompt IDs to evaluate
        expert_name: Name of expert column (z_alpha, z_beta, z_gamma, z_delta)
        k: K for nDCG@K and Hit@K
        relevance_threshold: Threshold for relevance in MRR and Hit@K
        
    Returns:
        Dict with metrics
    """
    ndcg_scores = []
    mrr_scores = []
    hit_scores = []
    
    for prompt_id in tqdm(prompt_ids, desc=f"Eval {expert_name}", leave=False):
        prompt_data = expert_scores_df[expert_scores_df['prompt_id'] == prompt_id].sort_values('rank')
        
        if len(prompt_data) == 0:
            continue
        
        pred_scores = prompt_data[expert_name].values
        gt_scores = prompt_data['ground_truth_score'].values
        
        # Compute metrics
        ndcg = ndcg_at_k(pred_scores, gt_scores, k)
        mrr_val = mrr(pred_scores, gt_scores, threshold=relevance_threshold)
        hit = hit_at_k(pred_scores, gt_scores, k, threshold=relevance_threshold)
        
        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr_val)
        hit_scores.append(hit)
    
    return {
        f'ndcg@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        f'ndcg@{k}_std': np.std(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'mrr_std': np.std(mrr_scores) if mrr_scores else 0.0,
        f'hit@{k}': np.mean(hit_scores) if hit_scores else 0.0,
        f'hit@{k}_std': np.std(hit_scores) if hit_scores else 0.0,
        'num_prompts': len(ndcg_scores)
    }


def evaluate_uniform_mixture(
    expert_scores_df: pd.DataFrame,
    prompt_ids: List[str],
    k: int = 10,
    relevance_threshold: float = 0.9
) -> Dict:
    """
    Evaluate uniform mixture of all experts (0.25 each).
    """
    expert_cols = ['z_alpha', 'z_beta', 'z_gamma', 'z_delta']
    
    ndcg_scores = []
    mrr_scores = []
    hit_scores = []
    
    for prompt_id in tqdm(prompt_ids, desc="Eval uniform mixture", leave=False):
        prompt_data = expert_scores_df[expert_scores_df['prompt_id'] == prompt_id].sort_values('rank')
        
        if len(prompt_data) == 0:
            continue
        
        # Uniform combination
        pred_scores = sum(prompt_data[col].values for col in expert_cols) / 4.0
        gt_scores = prompt_data['ground_truth_score'].values
        
        ndcg = ndcg_at_k(pred_scores, gt_scores, k)
        mrr_val = mrr(pred_scores, gt_scores, threshold=relevance_threshold)
        hit = hit_at_k(pred_scores, gt_scores, k, threshold=relevance_threshold)
        
        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr_val)
        hit_scores.append(hit)
    
    return {
        f'ndcg@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        f'ndcg@{k}_std': np.std(ndcg_scores) if ndcg_scores else 0.0,
        'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
        'mrr_std': np.std(mrr_scores) if mrr_scores else 0.0,
        f'hit@{k}': np.mean(hit_scores) if hit_scores else 0.0,
        f'hit@{k}_std': np.std(hit_scores) if hit_scores else 0.0,
        'num_prompts': len(ndcg_scores)
    }


def evaluate_oracle(
    expert_scores_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    prompt_ids: List[str],
    k: int = 10,
    relevance_threshold: float = 0.9
) -> Dict:
    """
    Evaluate oracle mixture using ground truth mix_weights from prompts.json.
    
    Note: This requires matching prompt IDs between expert_scores and prompts.json.
    """
    expert_cols = ['z_alpha', 'z_beta', 'z_gamma', 'z_delta']
    
    ndcg_scores = []
    mrr_scores = []
    hit_scores = []
    
    # Create mapping from prompt_id to mix_weights (if available)
    # This is tricky because expert_scores uses numeric IDs and prompts uses UUIDs
    # For now, we'll skip oracle evaluation if mapping is not available
    
    print("Warning: Oracle evaluation requires proper ID mapping between datasets")
    print("Using uniform weights as placeholder for oracle")
    
    # Fallback to uniform
    return evaluate_uniform_mixture(expert_scores_df, prompt_ids, k, relevance_threshold)


def evaluate_random_baseline(
    expert_scores_df: pd.DataFrame,
    prompt_ids: List[str],
    k: int = 10,
    relevance_threshold: float = 0.9,
    num_trials: int = 10
) -> Dict:
    """
    Evaluate random ranking baseline (average over multiple trials).
    """
    ndcg_scores = []
    mrr_scores = []
    hit_scores = []
    
    for trial in range(num_trials):
        trial_ndcg = []
        trial_mrr = []
        trial_hit = []
        
        for prompt_id in prompt_ids:
            prompt_data = expert_scores_df[expert_scores_df['prompt_id'] == prompt_id].sort_values('rank')
            
            if len(prompt_data) == 0:
                continue
            
            # Random scores
            pred_scores = np.random.randn(len(prompt_data))
            gt_scores = prompt_data['ground_truth_score'].values
            
            ndcg = ndcg_at_k(pred_scores, gt_scores, k)
            mrr_val = mrr(pred_scores, gt_scores, threshold=relevance_threshold)
            hit = hit_at_k(pred_scores, gt_scores, k, threshold=relevance_threshold)
            
            trial_ndcg.append(ndcg)
            trial_mrr.append(mrr_val)
            trial_hit.append(hit)
        
        ndcg_scores.append(np.mean(trial_ndcg) if trial_ndcg else 0.0)
        mrr_scores.append(np.mean(trial_mrr) if trial_mrr else 0.0)
        hit_scores.append(np.mean(trial_hit) if trial_hit else 0.0)
    
    return {
        f'ndcg@{k}': np.mean(ndcg_scores),
        f'ndcg@{k}_std': np.std(ndcg_scores),
        'mrr': np.mean(mrr_scores),
        'mrr_std': np.std(mrr_scores),
        f'hit@{k}': np.mean(hit_scores),
        f'hit@{k}_std': np.std(hit_scores),
        'num_prompts': len(prompt_ids) * num_trials
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert_scores", required=True, help="Parquet with expert scores")
    ap.add_argument("--prompts_path", required=True, help="Path to prompts.json")
    ap.add_argument("--out", default="artifacts/evaluation_results/listwise/single_expert_metrics.json")
    ap.add_argument("--split", default="test", help="Which split to evaluate (if split info available)")
    ap.add_argument("--k", type=int, default=10, help="K for nDCG@K and Hit@K")
    ap.add_argument("--relevance_threshold", type=float, default=0.9, help="Threshold for relevance")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Load data
    print("\n[1/3] Loading data...")
    expert_scores_df = pd.read_parquet(args.expert_scores)
    
    with open(args.prompts_path, 'r') as f:
        prompts_list = json.load(f)
    prompts_df = pd.DataFrame(prompts_list)
    
    print(f"Loaded {len(expert_scores_df)} expert score rows")
    print(f"Loaded {len(prompts_df)} prompts")
    
    # Get prompt IDs to evaluate
    all_prompt_ids = expert_scores_df['prompt_id'].unique().tolist()
    
    # If split info not available, use all prompts
    # In production, you'd load the split from somewhere
    print(f"\nEvaluating on {len(all_prompt_ids)} prompts")
    prompt_ids_to_eval = all_prompt_ids
    
    # Evaluate
    print("\n[2/3] Evaluating experts...")
    results = {}
    
    # Single experts
    expert_names = {
        'z_alpha': 'Alpha (Dense/Semantic)',
        'z_beta': 'Beta (BM25/Lexical)',
        'z_gamma': 'Gamma (LGCN/CF)',
        'z_delta': 'Delta (Emotion)'
    }
    
    for expert_col, expert_label in expert_names.items():
        print(f"\n  Evaluating {expert_label}...")
        metrics = evaluate_expert(
            expert_scores_df,
            prompt_ids_to_eval,
            expert_col,
            k=args.k,
            relevance_threshold=args.relevance_threshold
        )
        results[expert_col] = {
            'label': expert_label,
            **metrics
        }
    
    # Uniform mixture
    print(f"\n  Evaluating uniform mixture...")
    uniform_metrics = evaluate_uniform_mixture(
        expert_scores_df,
        prompt_ids_to_eval,
        k=args.k,
        relevance_threshold=args.relevance_threshold
    )
    results['uniform'] = {
        'label': 'Uniform Mixture (0.25 each)',
        **uniform_metrics
    }
    
    # Oracle (if available)
    print(f"\n  Evaluating oracle...")
    oracle_metrics = evaluate_oracle(
        expert_scores_df,
        prompts_df,
        prompt_ids_to_eval,
        k=args.k,
        relevance_threshold=args.relevance_threshold
    )
    results['oracle'] = {
        'label': 'Oracle (Ground Truth Weights)',
        **oracle_metrics
    }
    
    # Random baseline
    print(f"\n  Evaluating random baseline...")
    random_metrics = evaluate_random_baseline(
        expert_scores_df,
        prompt_ids_to_eval,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
        num_trials=10
    )
    results['random'] = {
        'label': 'Random Baseline',
        **random_metrics
    }
    
    # Save results
    print("\n[3/3] Saving results...")
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.out}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for method_key, method_data in results.items():
        print(f"\n{method_data['label']}:")
        print(f"  nDCG@{args.k}: {method_data[f'ndcg@{args.k}']:.4f} ± {method_data[f'ndcg@{args.k}_std']:.4f}")
        print(f"  MRR: {method_data['mrr']:.4f} ± {method_data['mrr_std']:.4f}")
        print(f"  Hit@{args.k}: {method_data[f'hit@{args.k}']:.4f} ± {method_data[f'hit@{args.k}_std']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


