#!/usr/bin/env python
"""Visualization utilities for evaluation results.

Generate publication-ready plots including:
- Performance comparison bar charts
- Category breakdown heatmaps
- Expert weight distributions
- Confusion matrices
- Error analysis plots
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def setup_plot_style():
    """Configure matplotlib/seaborn for publication-ready plots."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.dpi'] = 300


def plot_performance_comparison(
    results: Dict[str, Dict],
    metric: str = "agree_ties_0p5",
    output_path: Optional[Path] = None,
    title: str = "Model Performance Comparison"
):
    """Bar chart comparing model performance.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metric: Which metric to plot (agree_ties_0p5, agree_no_ties)
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    models = []
    scores = []
    
    for model_name, metrics in results.items():
        if "overall" in metrics and metric in metrics["overall"]:
            models.append(model_name)
            scores.append(metrics["overall"][metric])
    
    # Sort by score descending
    sorted_pairs = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    models, scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(models)), scores, color=sns.color_palette("viridis", len(models)))
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_category_heatmap(
    results: Dict[str, Dict],
    metric: str = "agree_ties_0p5",
    output_path: Optional[Path] = None,
    title: str = "Performance by Category"
):
    """Heatmap showing model performance across categories.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metric: Which metric to plot
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    # Build matrix: models x categories
    data_rows = []
    categories = set()
    
    for model_name, metrics in results.items():
        if "by_category" in metrics:
            for cat, cat_metrics in metrics["by_category"].items():
                categories.add(cat)
    
    categories = sorted(list(categories))
    
    for model_name in results.keys():
        row = []
        for cat in categories:
            if "by_category" in results[model_name]:
                cat_metrics = results[model_name]["by_category"].get(cat, {})
                row.append(cat_metrics.get(metric, np.nan))
            else:
                row.append(np.nan)
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows, index=list(results.keys()), columns=categories)
    
    fig, ax = plt.subplots(figsize=(max(10, len(categories)*1.5), max(6, len(results)*0.8)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.0, vmax=1.0,
                cbar_kws={'label': metric.replace('_', ' ').title()}, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_difficulty_heatmap(
    results: Dict[str, Dict],
    metric: str = "agree_ties_0p5",
    output_path: Optional[Path] = None,
    title: str = "Performance by Difficulty"
):
    """Heatmap showing model performance across difficulty levels.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metric: Which metric to plot
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    difficulties = ["easy", "medium", "hard"]
    data_rows = []
    
    for model_name, metrics in results.items():
        row = []
        for diff in difficulties:
            if "by_difficulty" in metrics:
                diff_metrics = metrics["by_difficulty"].get(diff, {})
                row.append(diff_metrics.get(metric, np.nan))
            else:
                row.append(np.nan)
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows, index=list(results.keys()), columns=difficulties)
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(results)*0.8)))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.0, vmax=1.0,
                cbar_kws={'label': metric.replace('_', ' ').title()}, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.yticks(rotation=0)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved difficulty heatmap to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_weight_distributions(
    weights_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Expert Weight Distributions"
):
    """Box plot or violin plot of expert weights.
    
    Args:
        weights_df: DataFrame with columns [alpha, beta, gamma, delta, category]
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    expert_cols = [col for col in ['alpha', 'beta', 'gamma', 'delta'] if col in weights_df.columns]
    
    if not expert_cols:
        print("No expert weight columns found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall distribution
    ax = axes[0]
    data_to_plot = [weights_df[col].dropna() for col in expert_cols]
    bp = ax.boxplot(data_to_plot, labels=expert_cols, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], sns.color_palette("Set2", len(expert_cols))):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Overall Weight Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Uniform (0.25)')
    ax.legend()
    
    # By category (if available)
    ax = axes[1]
    if 'category' in weights_df.columns:
        melted = weights_df.melt(id_vars=['category'], value_vars=expert_cols,
                                  var_name='Expert', value_name='Weight')
        sns.violinplot(data=melted, x='Expert', y='Weight', hue='category', ax=ax, split=False)
        ax.set_title('Weight Distribution by Category', fontsize=14, fontweight='bold')
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Just show violin plot overall
        melted = weights_df.melt(value_vars=expert_cols, var_name='Expert', value_name='Weight')
        sns.violinplot(data=melted, x='Expert', y='Weight', ax=ax)
        ax.set_title('Weight Distribution (Violin)', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Weight', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Uniform')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved weight distributions to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
):
    """Plot confusion matrix for classification tasks.
    
    Args:
        cm: Confusion matrix array (n_classes x n_classes)
        class_names: List of class names
        output_path: Where to save plot
        title: Plot title
        normalize: Whether to normalize by row (true labels)
    """
    setup_plot_style()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
                ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_feature_correlations(
    features_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Expert Feature Correlations"
):
    """Correlation heatmap for expert features (Δz).
    
    Args:
        features_df: DataFrame with dz_alpha, dz_beta, dz_gamma, dz_delta
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    feature_cols = [col for col in ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta'] 
                    if col in features_df.columns]
    
    if not feature_cols:
        print("No feature columns found")
        return
    
    corr = features_df[feature_cols].corr()
    
    # Rename for display
    display_names = {
        'dz_alpha': 'Dense (α)',
        'dz_beta': 'BM25 (β)',
        'dz_gamma': 'LightGCN (γ)',
        'dz_delta': 'Emotion (δ)'
    }
    corr = corr.rename(index=display_names, columns=display_names)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved correlation plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str],
    output_path: Optional[Path] = None,
    title: str = "Metric Comparison Across Models"
):
    """Grouped bar chart comparing multiple metrics across models.
    
    Args:
        comparison_df: DataFrame with models as index, metrics as columns
        metrics: List of metric column names to plot
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    if not available_metrics:
        print(f"No metrics found in dataframe. Available: {comparison_df.columns.tolist()}")
        return
    
    df_subset = comparison_df[available_metrics].copy()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    df_subset.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim([0, 1.0])
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved metric comparison to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_analysis(
    error_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Error Analysis by Category"
):
    """Stacked bar chart showing correct/incorrect/tie breakdown.
    
    Args:
        error_df: DataFrame with columns [category, correct, incorrect, ties]
        output_path: Where to save plot
        title: Plot title
    """
    setup_plot_style()
    
    if 'category' not in error_df.columns:
        print("Error dataframe must have 'category' column")
        return
    
    needed = ['correct', 'incorrect', 'ties']
    if not all(col in error_df.columns for col in needed):
        # Try alternate names
        alt_mapping = {'+1': 'correct', '-1': 'incorrect', '0(ties)': 'ties'}
        error_df = error_df.rename(columns=alt_mapping)
    
    plot_cols = [col for col in needed if col in error_df.columns]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    error_df.set_index('category')[plot_cols].plot(kind='bar', stacked=True, ax=ax,
                                                     color=['green', 'red', 'gray'])
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(title='Outcome')
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved error analysis to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_distribution_histogram(
    data: pd.Series,
    output_path: Optional[Path] = None,
    title: str = "Distribution",
    xlabel: str = "Value",
    bins: int = 30
):
    """Histogram with KDE overlay.
    
    Args:
        data: Series of values to plot
        output_path: Where to save plot
        title: Plot title
        xlabel: X-axis label
        bins: Number of histogram bins
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data.dropna(), bins=bins, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    
    # Add KDE
    from scipy.stats import gaussian_kde
    if len(data.dropna()) > 1:
        kde = gaussian_kde(data.dropna())
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {data.mean():.3f}\nStd: {data.std():.3f}\nMedian: {data.median():.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if output_path:
        plt.savefig(output_path)
        print(f"Saved histogram to {output_path}")
    else:
        plt.show()
    
    plt.close()

