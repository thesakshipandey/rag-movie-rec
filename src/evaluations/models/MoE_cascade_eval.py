#!/usr/bin/env python
"""Evaluate CASCADE-trained MoE routers across different thresholds.

This script evaluates multiple router models trained with different cascade
thresholds and analyzes the impact on expert selection and performance.

Example:
  python -m src.evaluations.models.MoE_cascade_eval \
    --features artifacts/router/features_sum.with_splits.bal.parquet \
    --output_dir artifacts/evaluation_results/cascade_training \
    --split test
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from src.router.mlp_router import RouterMLP
from src.cli.train_router import _prepare_feature_matrix

# Expert names
EXPERT_NAMES = ["alpha", "beta", "gamma", "delta"]


def pair_agreement_from_margin(s: torch.Tensor, y01: torch.Tensor, tol: float):
    """Convert margin scores to agreement labels."""
    j = torch.where(y01 > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    sign_s = torch.where(s.abs() <= tol, torch.zeros_like(s), torch.sign(s))
    return (sign_s * j).to(torch.int8)


def summarize_agreement(agree_np: np.ndarray) -> Dict:
    """Compute agreement statistics."""
    pos = int((agree_np == 1).sum())
    neg = int((agree_np == -1).sum())
    ties = int((agree_np == 0).sum())
    N = int(agree_np.size)
    return {
        "correct": pos,
        "incorrect": neg,
        "ties": ties,
        "total": N,
        "acc_no_ties": pos / max(1, pos + neg),
        "acc_ties_half": (pos + 0.5 * ties) / max(1, N)
    }


def analyze_expert_weights(weights: np.ndarray, thresholds: List[float]) -> Dict:
    """
    Analyze expert weight distributions and cascade behavior.
    
    Args:
        weights: [N, 4] array of expert weights
        thresholds: list of cascade thresholds to analyze
    
    Returns:
        Dictionary with weight statistics
    """
    max_weights = weights.max(axis=1)  # [N]
    dominant_expert_idx = weights.argmax(axis=1)  # [N]
    
    # Statistics per threshold
    threshold_stats = {}
    for threshold in thresholds:
        dominant_mask = max_weights >= threshold
        n_dominant = int(dominant_mask.sum())
        pct_dominant = 100.0 * n_dominant / len(weights)
        
        # Which expert dominates when gated
        if n_dominant > 0:
            gated_expert_counts = pd.Series(dominant_expert_idx[dominant_mask]).value_counts()
            gated_expert_dist = {
                EXPERT_NAMES[idx]: int(count) 
                for idx, count in gated_expert_counts.items()
            }
        else:
            gated_expert_dist = {}
        
        threshold_stats[f"threshold_{threshold:.2f}"] = {
            "dominant_count": n_dominant,
            "dominant_percentage": round(pct_dominant, 2),
            "gated_expert_distribution": gated_expert_dist
        }
    
    # Overall weight statistics
    mean_weights = weights.mean(axis=0)
    std_weights = weights.std(axis=0)
    
    return {
        "mean_weights": {EXPERT_NAMES[i]: float(mean_weights[i]) for i in range(4)},
        "std_weights": {EXPERT_NAMES[i]: float(std_weights[i]) for i in range(4)},
        "max_weight_mean": float(max_weights.mean()),
        "max_weight_std": float(max_weights.std()),
        "threshold_analysis": threshold_stats
    }


def evaluate_cascade_model(
    model_path: Path,
    features_df: pd.DataFrame,
    tie_tol: float = 0.05,
    thresholds: List[float] = [0.7, 0.75, 0.8, 0.85, 0.9]
) -> Dict:
    """
    Evaluate a single cascade-trained router model.
    
    Returns:
        Dictionary with evaluation results
    """
    # Prepare features
    X_np, y_np, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(features_df)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    
    # Load model
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Handle both old format (just state_dict) and new format (dict with metadata)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_config = checkpoint.get("config", {})
        trained_threshold = checkpoint.get("cascade_threshold", None)
        gating_enabled = checkpoint.get("gating_enabled", False)
    else:
        state_dict = checkpoint
        model_config = {}
        trained_threshold = None
        gating_enabled = False
    
    # Create model
    model = RouterMLP(d_in=X.shape[1], dz_dim=dz_dim, mix_indices=mix_indices or None)
    
    # Try to load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"  Warning: Could not load model completely: {e}")
        # Attempt partial load
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items() 
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    model.eval()
    
    # Inference
    with torch.no_grad():
        s, w = model(X)
        agree = pair_agreement_from_margin(s, y, tie_tol).cpu().numpy()
        weights = w.cpu().numpy()  # [N, 4]
    
    # Overall performance
    overall_stats = summarize_agreement(agree)
    
    # Weight analysis
    weight_stats = analyze_expert_weights(weights, thresholds)
    
    # Per-category breakdown
    category_stats = {}
    if "category" in features_df.columns:
        for cat in features_df["category"].unique():
            mask = features_df["category"] == cat
            cat_agree = agree[mask]
            if len(cat_agree) > 0:
                category_stats[str(cat)] = summarize_agreement(cat_agree)
    
    # Per-difficulty breakdown
    difficulty_stats = {}
    if "difficulty" in features_df.columns:
        for diff in features_df["difficulty"].unique():
            mask = features_df["difficulty"] == diff
            diff_agree = agree[mask]
            if len(diff_agree) > 0:
                difficulty_stats[str(diff)] = summarize_agreement(diff_agree)
    
    return {
        "model_path": str(model_path),
        "trained_threshold": trained_threshold,
        "gating_enabled": gating_enabled,
        "model_config": model_config,
        "overall": overall_stats,
        "weights": weight_stats,
        "by_category": category_stats,
        "by_difficulty": difficulty_stats
    }


def compare_cascade_models(results: List[Dict], output_dir: Path):
    """Generate comparison tables and reports."""
    
    # Create comparison dataframe
    comparison_data = []
    for result in results:
        model_name = Path(result["model_path"]).stem
        threshold = result.get("trained_threshold", "N/A")
        gating = result.get("gating_enabled", False)
        
        overall = result["overall"]
        weights = result["weights"]
        
        row = {
            "model": model_name,
            "trained_threshold": threshold if threshold != "N/A" else "none",
            "gating_enabled": gating,
            "accuracy_no_ties": round(overall["acc_no_ties"], 4),
            "accuracy_ties_half": round(overall["acc_ties_half"], 4),
            "correct": overall["correct"],
            "incorrect": overall["incorrect"],
            "ties": overall["ties"],
            "mean_max_weight": round(weights["max_weight_mean"], 4),
            "mean_alpha": round(weights["mean_weights"]["alpha"], 4),
            "mean_beta": round(weights["mean_weights"]["beta"], 4),
            "mean_gamma": round(weights["mean_weights"]["gamma"], 4),
            "mean_delta": round(weights["mean_weights"]["delta"], 4)
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by trained threshold (no gating first, then by threshold)
    comparison_df["sort_key"] = comparison_df["trained_threshold"].apply(
        lambda x: -1 if x == "none" else float(x)
    )
    comparison_df = comparison_df.sort_values("sort_key").drop("sort_key", axis=1)
    
    # Save comparison CSV
    csv_path = output_dir / "cascade_training_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved comparison table: {csv_path}")
    
    return comparison_df


def generate_summary_report(
    results: List[Dict], 
    comparison_df: pd.DataFrame, 
    output_dir: Path
):
    """Generate markdown summary report."""
    
    report_lines = [
        "# Cascade Training Evaluation Report",
        "",
        "## Overview",
        "",
        f"This report compares {len(results)} router models trained with different cascade configurations.",
        "",
        "## Overall Comparison",
        "",
        "### Performance Metrics",
        "",
        comparison_df[["model", "trained_threshold", "gating_enabled", 
                       "accuracy_no_ties", "accuracy_ties_half"]].to_markdown(index=False),
        "",
        "### Expert Weight Distributions",
        "",
        comparison_df[["model", "mean_alpha", "mean_beta", "mean_gamma", "mean_delta", 
                       "mean_max_weight"]].to_markdown(index=False),
        "",
        "## Detailed Analysis by Model",
        ""
    ]
    
    for result in results:
        model_name = Path(result["model_path"]).stem
        threshold = result.get("trained_threshold", "N/A")
        gating = result.get("gating_enabled", False)
        
        report_lines.extend([
            f"### {model_name}",
            "",
            f"- **Trained Threshold**: {threshold}",
            f"- **Gating Enabled**: {gating}",
            f"- **Accuracy (no ties)**: {result['overall']['acc_no_ties']:.4f}",
            f"- **Accuracy (ties=0.5)**: {result['overall']['acc_ties_half']:.4f}",
            ""
        ])
        
        # Cascade behavior at different thresholds
        if "threshold_analysis" in result["weights"]:
            report_lines.extend([
                "#### Cascade Behavior at Inference",
                ""
            ])
            
            threshold_data = []
            for thresh_key, thresh_stats in result["weights"]["threshold_analysis"].items():
                threshold_val = thresh_key.replace("threshold_", "")
                threshold_data.append({
                    "threshold": threshold_val,
                    "dominant_%": thresh_stats["dominant_percentage"],
                    "dominant_count": thresh_stats["dominant_count"]
                })
            
            if threshold_data:
                thresh_df = pd.DataFrame(threshold_data)
                report_lines.append(thresh_df.to_markdown(index=False))
                report_lines.append("")
        
        # Category breakdown
        if result.get("by_category"):
            report_lines.extend([
                "#### Performance by Category",
                ""
            ])
            cat_data = []
            for cat, stats in result["by_category"].items():
                cat_data.append({
                    "category": cat,
                    "accuracy": round(stats["acc_no_ties"], 4),
                    "count": stats["total"]
                })
            if cat_data:
                cat_df = pd.DataFrame(cat_data)
                report_lines.append(cat_df.to_markdown(index=False))
                report_lines.append("")
        
        report_lines.append("")
    
    # Best model recommendation
    best_no_ties = comparison_df.loc[comparison_df["accuracy_no_ties"].idxmax()]
    best_ties_half = comparison_df.loc[comparison_df["accuracy_ties_half"].idxmax()]
    
    report_lines.extend([
        "## Recommendations",
        "",
        f"- **Best Model (no ties)**: `{best_no_ties['model']}` with accuracy {best_no_ties['accuracy_no_ties']:.4f}",
        f"- **Best Model (ties=0.5)**: `{best_ties_half['model']}` with accuracy {best_ties_half['accuracy_ties_half']:.4f}",
        ""
    ])
    
    # Impact analysis
    if len(results) > 1:
        gated_models = comparison_df[comparison_df["gating_enabled"] == True]
        no_gate_models = comparison_df[comparison_df["gating_enabled"] == False]
        
        if len(gated_models) > 0 and len(no_gate_models) > 0:
            avg_gated_acc = gated_models["accuracy_no_ties"].mean()
            avg_no_gate_acc = no_gate_models["accuracy_no_ties"].mean()
            
            report_lines.extend([
                "## Impact of Cascade Gating",
                "",
                f"- **Average accuracy with gating**: {avg_gated_acc:.4f}",
                f"- **Average accuracy without gating**: {avg_no_gate_acc:.4f}",
                f"- **Difference**: {(avg_gated_acc - avg_no_gate_acc):.4f}",
                ""
            ])
    
    # Write report
    report_path = output_dir / "CASCADE_TRAINING_SUMMARY.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"Saved summary report: {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, 
                    help="Parquet file with features")
    ap.add_argument("--models", nargs="+", 
                    help="List of model paths to evaluate. If not provided, searches output_dir.")
    ap.add_argument("--output_dir", default="artifacts/evaluation_results/cascade_training",
                    help="Directory to save evaluation results")
    ap.add_argument("--split", default="test", choices=["all", "train", "val", "test"],
                    help="Which data split to evaluate on")
    ap.add_argument("--tie_tol", type=float, default=0.05,
                    help="Tolerance for tie decisions")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.7, 0.75, 0.8, 0.85, 0.9],
                    help="Cascade thresholds to analyze")
    
    args = ap.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features}")
    df = pd.read_parquet(args.features)
    
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("No 'split' column; create persistent splits first.")
        df = df[df["split"] == args.split].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows in split={args.split}")
    
    print(f"Evaluating on {len(df)} samples from split={args.split}")
    
    # Find models to evaluate
    if args.models:
        model_paths = [Path(m) for m in args.models]
    else:
        # Search for cascade models
        router_dir = Path("artifacts/router")
        model_paths = []
        
        # Look for cascade models
        for pattern in ["router_cascade_*.pt", "router_no_gating.pt"]:
            model_paths.extend(router_dir.glob(pattern))
        
        if not model_paths:
            print("No cascade models found. Please train models first or specify --models.")
            return
    
    model_paths = sorted(model_paths)
    print(f"\nFound {len(model_paths)} models to evaluate:")
    for p in model_paths:
        print(f"  - {p}")
    print()
    
    # Evaluate each model
    results = []
    for model_path in model_paths:
        print(f"Evaluating {model_path.name}...")
        try:
            result = evaluate_cascade_model(
                model_path, df, 
                tie_tol=args.tie_tol,
                thresholds=args.thresholds
            )
            results.append(result)
            
            # Save individual result
            result_file = output_dir / f"{model_path.stem}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {result_file}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    if not results:
        print("No models were successfully evaluated.")
        return
    
    # Generate comparison and report
    print("\nGenerating comparison and summary report...")
    comparison_df = compare_cascade_models(results, output_dir)
    generate_summary_report(results, comparison_df, output_dir)
    
    print("\n" + "=" * 80)
    print("CASCADE TRAINING EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()

