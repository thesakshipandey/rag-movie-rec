#!/usr/bin/env python
"""
Compare all methods: single experts, uniform, oracle, and trained router.

Generates comparison tables and plots for comprehensive analysis.

Usage:
    python -m src.evaluations.compare_methods \
        --single_expert_metrics artifacts/evaluation_results/listwise/single_expert_metrics.json \
        --router_metrics artifacts/evaluation_results/listwise/router_metrics.json \
        --out_dir artifacts/evaluation_results/listwise/comparison
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_metrics(single_expert_path: str, router_path: str) -> Dict:
    """Load all metrics from JSON files."""
    with open(single_expert_path, 'r') as f:
        single_expert = json.load(f)
    
    with open(router_path, 'r') as f:
        router = json.load(f)
    
    return {
        'single_expert': single_expert,
        'router': router
    }


def create_comparison_table(metrics: Dict, k: int = 10) -> pd.DataFrame:
    """Create comparison table of all methods."""
    rows = []
    
    # Single experts
    single_expert = metrics['single_expert']
    for method_key in ['z_alpha', 'z_beta', 'z_gamma', 'z_delta']:
        if method_key in single_expert:
            data = single_expert[method_key]
            rows.append({
                'Method': data['label'],
                'Type': 'Single Expert',
                f'nDCG@{k}': data[f'ndcg@{k}'],
                f'nDCG@{k} Std': data[f'ndcg@{k}_std'],
                'MRR': data['mrr'],
                'MRR Std': data['mrr_std'],
                f'Hit@{k}': data[f'hit@{k}'],
                f'Hit@{k} Std': data[f'hit@{k}_std'],
            })
    
    # Baselines
    for method_key in ['uniform', 'oracle', 'random']:
        if method_key in single_expert:
            data = single_expert[method_key]
            rows.append({
                'Method': data['label'],
                'Type': 'Baseline',
                f'nDCG@{k}': data[f'ndcg@{k}'],
                f'nDCG@{k} Std': data[f'ndcg@{k}_std'],
                'MRR': data['mrr'],
                'MRR Std': data['mrr_std'],
                f'Hit@{k}': data[f'hit@{k}'],
                f'Hit@{k} Std': data[f'hit@{k}_std'],
            })
    
    # Router
    router_data = metrics['router']['aggregate_metrics']
    rows.append({
        'Method': 'Contextual Hedge Router',
        'Type': 'Trained Router',
        f'nDCG@{k}': router_data[f'ndcg@{k}'],
        f'nDCG@{k} Std': router_data[f'ndcg@{k}_std'],
        'MRR': router_data['mrr'],
        'MRR Std': router_data['mrr_std'],
        f'Hit@{k}': router_data[f'hit@{k}'],
        f'Hit@{k} Std': router_data[f'hit@{k}_std'],
    })
    
    df = pd.DataFrame(rows)
    
    # Sort by nDCG@k descending
    df = df.sort_values(f'nDCG@{k}', ascending=False)
    
    return df


def plot_comparison_bar_chart(df: pd.DataFrame, out_path: str, k: int = 10):
    """Create bar chart comparing methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [f'nDCG@{k}', 'MRR', f'Hit@{k}']
    titles = [f'nDCG@{k}', 'MRR', f'Hit@{k}']
    
    for ax, metric, title in zip(axes, metrics, titles):
        # Sort by metric
        df_sorted = df.sort_values(metric, ascending=True)
        
        # Color by type
        colors = df_sorted['Type'].map({
            'Single Expert': 'steelblue',
            'Baseline': 'gray',
            'Trained Router': 'crimson'
        })
        
        ax.barh(df_sorted['Method'], df_sorted[metric], color=colors, alpha=0.8)
        ax.set_xlabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (val, err) in enumerate(zip(df_sorted[metric], df_sorted.get(f'{metric} Std', [0]*len(df_sorted)))):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison bar chart to {out_path}")
    plt.close()


def plot_expert_usage(metrics: Dict, out_path: str):
    """Plot expert usage by router."""
    router_data = metrics['router']['aggregate_metrics']
    expert_usage = router_data['expert_usage']
    
    expert_names = ['Alpha\n(Dense)', 'Beta\n(BM25)', 'Gamma\n(LGCN)', 'Delta\n(Emotion)']
    expert_weights = [expert_usage['alpha'], expert_usage['beta'], expert_usage['gamma'], expert_usage['delta']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.bar(expert_names, expert_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Average Weight', fontsize=14)
    ax.set_title('Expert Usage by Contextual Hedge Router', fontsize=16, fontweight='bold')
    ax.set_ylim(0, max(expert_weights) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, weight in zip(bars, expert_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add uniform baseline line
    ax.axhline(y=0.25, color='red', linestyle='--', linewidth=2, label='Uniform (0.25)', alpha=0.7)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved expert usage plot to {out_path}")
    plt.close()


def plot_expert_weight_distribution(metrics: Dict, out_path: str):
    """Plot distribution of expert weights across prompts."""
    expert_analysis = metrics['router']['expert_selection_analysis']
    weight_stats = expert_analysis['weight_statistics']
    
    expert_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    expert_keys = ['alpha', 'beta', 'gamma', 'delta']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for ax, name, key, color in zip(axes, expert_names, expert_keys, colors):
        stats = weight_stats[key]
        
        # Bar plot showing mean, min, max
        x_pos = [0, 1, 2]
        values = [stats['min'], stats['mean'], stats['max']]
        labels = ['Min', 'Mean', 'Max']
        
        bars = ax.bar(x_pos, values, color=[color]*3, alpha=[0.4, 1.0, 0.4], edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(f'{name} Expert\n(Dominant: {stats["dominant_count"]} times)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add std dev annotation
        ax.text(0.95, 0.95, f'σ = {stats["std"]:.3f}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved expert weight distribution plot to {out_path}")
    plt.close()


def plot_relative_improvement(df: pd.DataFrame, out_path: str, k: int = 10):
    """Plot relative improvement of router over baselines."""
    # Get router performance
    router_row = df[df['Method'] == 'Contextual Hedge Router'].iloc[0]
    
    # Get baselines
    baseline_rows = df[df['Type'] == 'Baseline']
    
    if len(baseline_rows) == 0:
        print("No baseline methods found for relative improvement plot")
        return
    
    metrics = [f'nDCG@{k}', 'MRR', f'Hit@{k}']
    
    improvements = []
    for _, baseline in baseline_rows.iterrows():
        row_data = {'Baseline': baseline['Method']}
        for metric in metrics:
            baseline_val = baseline[metric]
            router_val = router_row[metric]
            if baseline_val > 0:
                improvement = ((router_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0
            row_data[metric] = improvement
        improvements.append(row_data)
    
    imp_df = pd.DataFrame(improvements)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(imp_df))
    width = 0.25
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, imp_df[metric], width, label=metric, color=color, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 0.5,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    ax.set_xlabel('Baseline Method', fontsize=12)
    ax.set_ylabel('Relative Improvement (%)', fontsize=12)
    ax.set_title('Router Performance vs Baselines', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(imp_df['Baseline'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved relative improvement plot to {out_path}")
    plt.close()


def generate_latex_table(df: pd.DataFrame, out_path: str, k: int = 10):
    """Generate LaTeX table for paper."""
    # Format values
    df_latex = df.copy()
    
    # Create formatted columns with ±std
    df_latex[f'nDCG@{k}_formatted'] = df_latex.apply(
        lambda row: f"{row[f'nDCG@{k}']:.3f} ± {row[f'nDCG@{k} Std']:.3f}", axis=1
    )
    df_latex['MRR_formatted'] = df_latex.apply(
        lambda row: f"{row['MRR']:.3f} ± {row['MRR Std']:.3f}", axis=1
    )
    df_latex[f'Hit@{k}_formatted'] = df_latex.apply(
        lambda row: f"{row[f'Hit@{k}']:.3f} ± {row[f'Hit@{k} Std']:.3f}", axis=1
    )
    
    # Select columns for table
    table_df = df_latex[['Method', 'Type', f'nDCG@{k}_formatted', 'MRR_formatted', f'Hit@{k}_formatted']]
    table_df.columns = ['Method', 'Type', f'nDCG@{k}', 'MRR', f'Hit@{k}']
    
    # Generate LaTeX
    latex_str = table_df.to_latex(index=False, escape=False)
    
    with open(out_path, 'w') as f:
        f.write(latex_str)
    
    print(f"Saved LaTeX table to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_expert_metrics", required=True)
    ap.add_argument("--router_metrics", required=True)
    ap.add_argument("--out_dir", default="artifacts/evaluation_results/listwise/comparison")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load metrics
    print("\n[1/5] Loading metrics...")
    metrics = load_metrics(args.single_expert_metrics, args.router_metrics)
    
    # Create comparison table
    print("\n[2/5] Creating comparison table...")
    comparison_df = create_comparison_table(metrics, k=args.k)
    
    # Save table
    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved comparison table to {csv_path}")
    
    # Print table
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)
    
    # Generate plots
    print("\n[3/5] Generating comparison plots...")
    plot_comparison_bar_chart(
        comparison_df,
        os.path.join(args.out_dir, "comparison_bar_chart.png"),
        k=args.k
    )
    
    print("\n[4/5] Generating expert usage plots...")
    plot_expert_usage(
        metrics,
        os.path.join(args.out_dir, "expert_usage.png")
    )
    
    plot_expert_weight_distribution(
        metrics,
        os.path.join(args.out_dir, "expert_weight_distribution.png")
    )
    
    plot_relative_improvement(
        comparison_df,
        os.path.join(args.out_dir, "relative_improvement.png"),
        k=args.k
    )
    
    # Generate LaTeX table
    print("\n[5/5] Generating LaTeX table...")
    generate_latex_table(
        comparison_df,
        os.path.join(args.out_dir, "comparison_table.tex"),
        k=args.k
    )
    
    # Save summary
    summary = {
        'best_method': comparison_df.iloc[0]['Method'],
        'best_ndcg': float(comparison_df.iloc[0][f'nDCG@{args.k}']),
        'best_mrr': float(comparison_df.iloc[0]['MRR']),
        'best_hit': float(comparison_df.iloc[0][f'Hit@{args.k}']),
        'router_rank': int(comparison_df[comparison_df['Method'] == 'Contextual Hedge Router'].index[0] + 1),
        'total_methods': len(comparison_df)
    }
    
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nAll outputs saved to: {args.out_dir}")
    print(f"\nBest method: {summary['best_method']}")
    print(f"  nDCG@{args.k}: {summary['best_ndcg']:.4f}")
    print(f"  MRR: {summary['best_mrr']:.4f}")
    print(f"  Hit@{args.k}: {summary['best_hit']:.4f}")
    print(f"\nRouter rank: {summary['router_rank']}/{summary['total_methods']}")


if __name__ == "__main__":
    main()


