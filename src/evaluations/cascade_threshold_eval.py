#!/usr/bin/env python
"""Evaluate cascade routing with different dominance thresholds.

Tests the dominance gating mechanism with different thresholds:
- max(w) ≥ threshold triggers cascade filtering
- Compare: 0.7, 0.75, 0.8, 0.85, 0.9, and no gating

Usage:
    python -m src.evaluations.cascade_threshold_eval \
        --features artifacts/router/features_sum.with_splits.bal.parquet \
        --router_model artifacts/router/router_mlp_sum.pt \
        --output_dir artifacts/evaluation_results/cascade_analysis \
        --split test
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from src.router.mlp_router import RouterMLP
from src.cli.train_router import _prepare_feature_matrix


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float = 0.05):
    """Compute agreement on pairwise judgments."""
    j = torch.where(y01 > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    return (sign_s * j).to(torch.int8)


def summarize(agree_np: np.ndarray):
    """Summarize agreement metrics."""
    pos = int((agree_np == 1).sum())
    neg = int((agree_np == -1).sum())
    ties = int((agree_np == 0).sum())
    N = int(agree_np.size)
    acc_nt = pos / max(1, pos + neg)
    acc_ties = (pos + 0.5 * ties) / max(1, N)
    return pos, neg, ties, acc_nt, acc_ties


def check_dominance(weights: torch.Tensor, threshold: float = 0.75) -> tuple:
    """Check if one expert dominates with given threshold.
    
    Args:
        weights: [B, 4] expert weights
        threshold: Minimum weight for dominance
        
    Returns:
        (is_dominant, dominant_idx) for each sample
    """
    max_weights, max_indices = weights.max(dim=1)
    is_dominant = max_weights >= threshold
    return is_dominant, max_indices


def evaluate_with_cascade(
    features_df: pd.DataFrame,
    router_model: RouterMLP,
    threshold: float = 0.75,
    tie_tol: float = 0.05
) -> Dict[str, Any]:
    """Evaluate router with cascade gating at given threshold.
    
    Args:
        features_df: Features dataframe
        router_model: Trained router
        threshold: Dominance threshold (0.0 = no gating, >0 = gating)
        tie_tol: Tolerance for ties
        
    Returns:
        Dictionary with metrics and gating statistics
    """
    X_np, y_np, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(features_df)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    
    router_model.eval()
    
    with torch.no_grad():
        # Get router predictions and weights
        s, w = router_model(X)
        
        # Check dominance
        if threshold > 0:
            is_dominant, dominant_idx = check_dominance(w, threshold)
            gating_triggered = is_dominant.sum().item()
            gating_rate = gating_triggered / len(features_df)
        else:
            # No gating case
            is_dominant = torch.zeros(len(features_df), dtype=torch.bool)
            gating_triggered = 0
            gating_rate = 0.0
        
        # Compute agreement
        agree = pair_agreement_from_margin(s, y, tie_tol).cpu().numpy()
    
    pos, neg, ties, acc_nt, acc_ties = summarize(agree)
    
    results = {
        'threshold': threshold,
        'overall': {
            'correct': pos,
            'incorrect': neg,
            'ties': ties,
            'total': len(features_df),
            'agree_no_ties': float(acc_nt),
            'agree_ties_0p5': float(acc_ties)
        },
        'gating_stats': {
            'gating_triggered': gating_triggered,
            'gating_rate': float(gating_rate),
            'no_gating': len(features_df) - gating_triggered
        }
    }
    
    # Performance on gated vs non-gated samples
    if gating_triggered > 0:
        gated_agree = agree[is_dominant.cpu().numpy()]
        non_gated_agree = agree[~is_dominant.cpu().numpy()]
        
        p_g, n_g, t_g, a1_g, a2_g = summarize(gated_agree)
        p_ng, n_ng, t_ng, a1_ng, a2_ng = summarize(non_gated_agree)
        
        results['gated_samples'] = {
            'count': gating_triggered,
            'correct': p_g,
            'incorrect': n_g,
            'ties': t_g,
            'agree_no_ties': float(a1_g),
            'agree_ties_0p5': float(a2_g)
        }
        
        results['non_gated_samples'] = {
            'count': len(features_df) - gating_triggered,
            'correct': p_ng,
            'incorrect': n_ng,
            'ties': t_ng,
            'agree_no_ties': float(a1_ng),
            'agree_ties_0p5': float(a2_ng)
        }
    
    # Dominant expert distribution
    if gating_triggered > 0:
        dominant_experts = dominant_idx[is_dominant].cpu().numpy()
        expert_names = ['alpha', 'beta', 'gamma', 'delta']
        expert_counts = {}
        for i, name in enumerate(expert_names):
            expert_counts[name] = int((dominant_experts == i).sum())
        results['dominant_expert_distribution'] = expert_counts
    
    # Weight statistics
    weight_stats = {}
    expert_names = ['alpha', 'beta', 'gamma', 'delta']
    for i, name in enumerate(expert_names):
        expert_weights = w[:, i].cpu().numpy()
        weight_stats[name] = {
            'mean': float(expert_weights.mean()),
            'std': float(expert_weights.std()),
            'max': float(expert_weights.max()),
            'above_threshold': int((expert_weights >= threshold).sum())
        }
    results['weight_stats'] = weight_stats
    
    # By category analysis
    if 'category' in features_df.columns:
        results['by_category'] = {}
        for cat in features_df['category'].unique():
            cat_mask = features_df['category'] == cat
            cat_agree = agree[cat_mask]
            p, n, t, a1, a2 = summarize(cat_agree)
            
            cat_result = {
                'correct': p,
                'incorrect': n,
                'ties': t,
                'agree_no_ties': float(a1),
                'agree_ties_0p5': float(a2),
                'count': len(cat_agree)
            }
            
            if threshold > 0:
                cat_gated = is_dominant.cpu().numpy()[cat_mask]
                cat_result['gating_rate'] = float(cat_gated.mean())
            
            results['by_category'][cat] = cat_result
    
    return results


def run_cascade_threshold_analysis(
    features_path: Path,
    router_model_path: Path,
    output_dir: Path,
    split: str = "test",
    thresholds: List[float] = None
):
    """Run cascade analysis with multiple thresholds.
    
    Args:
        features_path: Path to features parquet
        router_model_path: Path to router model
        output_dir: Output directory
        split: Which split to evaluate
        thresholds: List of thresholds to test
    """
    if thresholds is None:
        thresholds = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CASCADE DOMINANCE THRESHOLD ANALYSIS")
    print("="*80)
    print(f"\nRouter model: {router_model_path}")
    print(f"Features: {features_path}")
    print(f"Split: {split}")
    print(f"Thresholds: {thresholds}")
    print("")
    
    # Load features
    print(f"Loading features...")
    features_df = pd.read_parquet(features_path)
    
    if split != "all":
        if "split" not in features_df.columns:
            raise ValueError("No 'split' column found in features")
        features_df = features_df[features_df["split"] == split].copy().reset_index(drop=True)
        if features_df.empty:
            raise ValueError(f"No rows for split={split}")
    
    print(f"Loaded {len(features_df)} pairs")
    
    # Load router model
    print(f"Loading router model...")
    X_np, y_np, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(features_df)
    
    state = torch.load(router_model_path, map_location="cpu")
    
    # Infer input dimension
    if 'net.0.weight' in state:
        saved_input_dim = state['net.0.weight'].shape[1]
    else:
        saved_input_dim = X_np.shape[1]
    
    router_model = RouterMLP(d_in=saved_input_dim, dz_dim=dz_dim, mix_indices=mix_indices or None)
    router_model.load_state_dict(state, strict=False)
    router_model.eval()
    print(f"Model loaded successfully")
    
    # Evaluate each threshold
    all_results = {}
    
    for threshold in thresholds:
        threshold_name = f"threshold_{threshold:.2f}" if threshold > 0 else "no_gating"
        print(f"\n[{threshold_name}] Evaluating with threshold={threshold}...")
        
        results = evaluate_with_cascade(
            features_df,
            router_model,
            threshold=threshold,
            tie_tol=0.05
        )
        
        all_results[threshold_name] = results
        
        # Print summary
        print(f"  Agreement (ties=0.5): {results['overall']['agree_ties_0p5']:.4f}")
        print(f"  Agreement (no ties):  {results['overall']['agree_no_ties']:.4f}")
        
        if threshold > 0:
            print(f"  Gating rate: {results['gating_stats']['gating_rate']*100:.1f}%")
            print(f"  Gated samples: {results['gating_stats']['gating_triggered']}")
            
            if 'gated_samples' in results:
                print(f"    - Gated performance: {results['gated_samples']['agree_ties_0p5']:.4f}")
                print(f"    - Non-gated performance: {results['non_gated_samples']['agree_ties_0p5']:.4f}")
            
            if 'dominant_expert_distribution' in results:
                print(f"  Dominant experts:")
                for expert, count in results['dominant_expert_distribution'].items():
                    print(f"    - {expert}: {count} ({count/results['gating_stats']['gating_triggered']*100:.1f}%)")
        
        # Save individual result
        result_file = output_dir / f"{threshold_name}_metrics.json"
        with open(result_file, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        print(f"  Saved to {result_file}")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print("="*80)
    
    comparison_data = []
    for threshold_name, results in all_results.items():
        row = {
            'threshold': results['threshold'],
            'threshold_name': threshold_name,
            'agree_ties_0p5': results['overall']['agree_ties_0p5'],
            'agree_no_ties': results['overall']['agree_no_ties'],
            'correct': results['overall']['correct'],
            'incorrect': results['overall']['incorrect'],
            'ties': results['overall']['ties']
        }
        
        if 'gating_stats' in results:
            row['gating_rate'] = results['gating_stats']['gating_rate']
            row['gating_triggered'] = results['gating_stats']['gating_triggered']
        
        if 'gated_samples' in results:
            row['gated_agree'] = results['gated_samples']['agree_ties_0p5']
            row['non_gated_agree'] = results['non_gated_samples']['agree_ties_0p5']
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('threshold')
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "threshold_comparison.csv", index=False)
    print(f"\nComparison table saved to {output_dir / 'threshold_comparison.csv'}")
    
    # Print comparison table
    print("\nPerformance by Threshold:")
    print(comparison_df[['threshold_name', 'agree_ties_0p5', 'gating_rate', 'gated_agree', 'non_gated_agree']].to_string(index=False))
    
    # Find best threshold
    best_idx = comparison_df['agree_ties_0p5'].idxmax()
    best_threshold = comparison_df.loc[best_idx]
    
    print(f"\n{'='*80}")
    print("BEST THRESHOLD")
    print("="*80)
    print(f"Threshold: {best_threshold['threshold_name']}")
    print(f"Agreement: {best_threshold['agree_ties_0p5']:.4f}")
    if 'gating_rate' in best_threshold:
        print(f"Gating rate: {best_threshold['gating_rate']*100:.1f}%")
    
    # Save complete results
    summary = {
        'thresholds_tested': thresholds,
        'best_threshold': {
            'name': best_threshold['threshold_name'],
            'value': float(best_threshold['threshold']),
            'agreement': float(best_threshold['agree_ties_0p5'])
        },
        'all_results': all_results
    }
    
    with open(output_dir / "cascade_analysis_summary.json", 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    print(f"\nComplete summary saved to {output_dir / 'cascade_analysis_summary.json'}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    return all_results, comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cascade routing with different dominance thresholds"
    )
    
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features parquet"
    )
    
    parser.add_argument(
        "--router_model",
        type=Path,
        required=True,
        help="Path to router model (.pt file)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to evaluate"
    )
    
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.7, 0.75, 0.8, 0.85, 0.9],
        help="Dominance thresholds to test (0.0 = no gating)"
    )
    
    args = parser.parse_args()
    
    run_cascade_threshold_analysis(
        features_path=args.features,
        router_model_path=args.router_model,
        output_dir=args.output_dir,
        split=args.split,
        thresholds=args.thresholds
    )


if __name__ == "__main__":
    main()

