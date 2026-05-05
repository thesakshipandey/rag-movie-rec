#!/usr/bin/env python
"""Comparison and ablation analysis for models.

Analyze:
- Model comparison across all metrics
- Expert ablation studies
- Weight analysis (learned vs uniform vs oracle)
- Category-wise and difficulty-wise performance
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
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
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def create_comparison_table(
    results: Dict[str, Dict],
    metrics: List[str] = None
) -> pd.DataFrame:
    """Create comparison table across all models.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metrics: List of metrics to compare (if None, use all available)
        
    Returns:
        DataFrame with models as rows, metrics as columns
    """
    if metrics is None:
        metrics = ['agree_no_ties', 'agree_ties_0p5']
    
    rows = []
    for model_name, model_results in results.items():
        row = {'model': model_name}
        
        if 'overall' in model_results:
            for metric in metrics:
                if metric in model_results['overall']:
                    row[metric] = model_results['overall'][metric]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if 'model' in df.columns:
        df = df.set_index('model')
    
    # Sort by primary metric (agree_ties_0p5 or first metric)
    sort_col = 'agree_ties_0p5' if 'agree_ties_0p5' in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False)
    
    return df


def analyze_by_category(
    results: Dict[str, Dict],
    metric: str = 'agree_ties_0p5'
) -> pd.DataFrame:
    """Create category-wise performance table.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metric: Which metric to analyze
        
    Returns:
        DataFrame with models as rows, categories as columns
    """
    # Collect all categories
    all_categories = set()
    for model_results in results.values():
        if 'by_category' in model_results:
            all_categories.update(model_results['by_category'].keys())
    
    all_categories = sorted(list(all_categories))
    
    rows = []
    for model_name, model_results in results.items():
        row = {'model': model_name}
        
        if 'by_category' in model_results:
            for cat in all_categories:
                if cat in model_results['by_category']:
                    row[cat] = model_results['by_category'][cat].get(metric, np.nan)
                else:
                    row[cat] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if 'model' in df.columns:
        df = df.set_index('model')
    
    return df


def analyze_by_difficulty(
    results: Dict[str, Dict],
    metric: str = 'agree_ties_0p5'
) -> pd.DataFrame:
    """Create difficulty-wise performance table.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metric: Which metric to analyze
        
    Returns:
        DataFrame with models as rows, difficulties as columns
    """
    difficulties = ['easy', 'medium', 'hard']
    
    rows = []
    for model_name, model_results in results.items():
        row = {'model': model_name}
        
        if 'by_difficulty' in model_results:
            for diff in difficulties:
                if diff in model_results['by_difficulty']:
                    row[diff] = model_results['by_difficulty'][diff].get(metric, np.nan)
                else:
                    row[diff] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if 'model' in df.columns:
        df = df.set_index('model')
    
    return df


def identify_best_models(
    results: Dict[str, Dict],
    metric: str = 'agree_ties_0p5'
) -> Dict[str, Any]:
    """Identify best performing models overall and by category/difficulty.
    
    Args:
        results: Dict mapping model_name -> metrics dict
        metric: Which metric to use for ranking
        
    Returns:
        Dict with best models for different scenarios
    """
    best = {}
    
    # Overall best
    overall_scores = {}
    for model_name, model_results in results.items():
        if 'overall' in model_results and metric in model_results['overall']:
            overall_scores[model_name] = model_results['overall'][metric]
    
    if overall_scores:
        best_overall = max(overall_scores.items(), key=lambda x: x[1])
        best['overall'] = {
            'model': best_overall[0],
            'score': best_overall[1]
        }
    
    # Best per category
    category_df = analyze_by_category(results, metric)
    best['by_category'] = {}
    for cat in category_df.columns:
        best_model = category_df[cat].idxmax()
        best_score = category_df[cat].max()
        if not pd.isna(best_score):
            best['by_category'][cat] = {
                'model': best_model,
                'score': float(best_score)
            }
    
    # Best per difficulty
    difficulty_df = analyze_by_difficulty(results, metric)
    best['by_difficulty'] = {}
    for diff in difficulty_df.columns:
        best_model = difficulty_df[diff].idxmax()
        best_score = difficulty_df[diff].max()
        if not pd.isna(best_score):
            best['by_difficulty'][diff] = {
                'model': best_model,
                'score': float(best_score)
            }
    
    # Most robust (smallest variance across categories)
    if not category_df.empty:
        variances = category_df.var(axis=1)
        most_robust = variances.idxmin()
        best['most_robust'] = {
            'model': most_robust,
            'variance': float(variances.min()),
            'mean_score': float(category_df.loc[most_robust].mean())
        }
    
    return best


def expert_ablation_analysis(
    expert_results: Dict[str, Dict],
    moe_results: Dict[str, Dict],
    metric: str = 'agree_ties_0p5'
) -> pd.DataFrame:
    """Analyze contribution of each expert.
    
    Compare:
    - Individual expert performance
    - MoE with all experts
    - Hypothetical MoE with each expert removed
    
    Args:
        expert_results: Dict mapping expert_name -> metrics
        moe_results: Dict mapping moe_model -> metrics
        metric: Which metric to analyze
        
    Returns:
        DataFrame with ablation results
    """
    rows = []
    
    # Individual experts
    for expert, results in expert_results.items():
        if 'overall' in results and metric in results['overall']:
            rows.append({
                'configuration': f'Expert_{expert}_only',
                'type': 'individual',
                'score': results['overall'][metric],
                'expert': expert
            })
    
    # Full MoE
    for moe_name, results in moe_results.items():
        if 'overall' in results and metric in results['overall']:
            rows.append({
                'configuration': moe_name,
                'type': 'full_moe',
                'score': results['overall'][metric],
                'expert': 'all'
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('score', ascending=False)
    
    return df


def weight_distribution_analysis(
    moe_results: Dict[str, Any],
    features_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Analyze expert weight distributions from MoE router.
    
    Args:
        moe_results: MoE evaluation results (may contain weight predictions)
        features_df: Features dataframe to extract weight info
        
    Returns:
        Dictionary with weight statistics
    """
    weight_stats = {}
    
    # If weights are stored in results
    if 'weights' in moe_results:
        weights = moe_results['weights']
        
        for expert in ['alpha', 'beta', 'gamma', 'delta']:
            if expert in weights:
                expert_weights = np.array(weights[expert])
                weight_stats[expert] = {
                    'mean': float(expert_weights.mean()),
                    'std': float(expert_weights.std()),
                    'min': float(expert_weights.min()),
                    'max': float(expert_weights.max()),
                    'median': float(np.median(expert_weights)),
                    'q25': float(np.percentile(expert_weights, 25)),
                    'q75': float(np.percentile(expert_weights, 75))
                }
    
    # Analyze from features if available
    # (Would need to re-run router to get per-sample weights)
    
    return weight_stats


def generate_comparison_summary(
    all_results: Dict[str, Dict],
    output_dir: Path,
    features_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Generate comprehensive comparison analysis.
    
    Args:
        all_results: Dict with keys 'experts', 'moe', 'ranknet', etc.
        output_dir: Where to save outputs
        features_df: Optional features dataframe for additional analysis
        
    Returns:
        Summary dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    
    # Flatten all results for unified comparison
    flat_results = {}
    for category, models in all_results.items():
        for model_name, metrics in models.items():
            flat_results[f"{category}_{model_name}"] = metrics
    
    # Overall comparison
    print("Creating overall comparison table...")
    comparison_df = create_comparison_table(flat_results)
    comparison_df.to_csv(output_dir / "all_models_comparison.csv")
    summary['overall_ranking'] = comparison_df.to_dict()
    print(f"Saved comparison to {output_dir / 'all_models_comparison.csv'}")
    
    # Category-wise performance
    print("Analyzing category-wise performance...")
    category_df = analyze_by_category(flat_results)
    category_df.to_csv(output_dir / "category_performance.csv")
    summary['category_performance'] = category_df.to_dict()
    print(f"Saved category analysis to {output_dir / 'category_performance.csv'}")
    
    # Difficulty-wise performance
    print("Analyzing difficulty-wise performance...")
    difficulty_df = analyze_by_difficulty(flat_results)
    difficulty_df.to_csv(output_dir / "difficulty_analysis.csv")
    summary['difficulty_performance'] = difficulty_df.to_dict()
    print(f"Saved difficulty analysis to {output_dir / 'difficulty_analysis.csv'}")
    
    # Identify best models
    print("Identifying best models...")
    best_models = identify_best_models(flat_results)
    summary['best_models'] = best_models
    with open(output_dir / "best_models.json", 'w') as f:
        json.dump(convert_to_serializable(best_models), f, indent=2)
    print(f"Saved best models to {output_dir / 'best_models.json'}")
    
    # Expert ablation
    if 'experts' in all_results and 'moe' in all_results:
        print("Performing expert ablation analysis...")
        ablation_df = expert_ablation_analysis(
            all_results['experts'],
            all_results['moe']
        )
        ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)
        summary['ablation'] = ablation_df.to_dict()
        print(f"Saved ablation results to {output_dir / 'ablation_results.csv'}")
    
    # Save complete summary
    with open(output_dir / "comparison_summary.json", 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    print(f"Saved complete summary to {output_dir / 'comparison_summary.json'}")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comparison analysis")
    parser.add_argument("--results_dir", required=True, help="Directory with evaluation results")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--features", help="Optional features parquet for additional analysis")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Load all results
    all_results = {}
    
    for subdir in ['experts', 'moe', 'ranknet']:
        subdir_path = results_dir / subdir
        if subdir_path.exists():
            all_results[subdir] = {}
            for json_file in subdir_path.glob("*_metrics.json"):
                model_name = json_file.stem.replace('_metrics', '')
                with open(json_file, 'r') as f:
                    all_results[subdir][model_name] = json.load(f)
    
    # Load features if provided
    features_df = None
    if args.features:
        features_df = pd.read_parquet(args.features)
    
    summary = generate_comparison_summary(all_results, output_dir, features_df)
    
    print("\n=== Comparison Analysis Complete ===")
    if 'best_models' in summary and 'overall' in summary['best_models']:
        best = summary['best_models']['overall']
        print(f"Best overall model: {best['model']} (score: {best['score']:.4f})")

