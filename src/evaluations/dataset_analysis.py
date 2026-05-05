#!/usr/bin/env python
"""Dataset analysis module for comprehensive statistics.

Analyze:
- Prompt distributions (category, difficulty, length)
- Pair distributions (preferences, judgments)
- Movie metadata distributions
- Expert feature distributions and correlations
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter


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


def analyze_prompts(
    features_df: pd.DataFrame,
    prompts_json: Optional[Path] = None
) -> Dict[str, Any]:
    """Analyze prompt distributions.
    
    Args:
        features_df: Features dataframe with prompt_id, category, difficulty
        prompts_json: Optional path to prompts.json for text analysis
        
    Returns:
        Dictionary with prompt statistics
    """
    stats = {}
    
    # Get unique prompts
    if 'prompt_id' in features_df.columns:
        prompt_df = features_df.groupby('prompt_id').first().reset_index()
        stats['total_prompts'] = len(prompt_df)
        
        # Category distribution
        if 'category' in prompt_df.columns:
            cat_dist = prompt_df['category'].value_counts().to_dict()
            stats['category_distribution'] = cat_dist
            stats['category_counts'] = dict(prompt_df['category'].value_counts())
        
        # Difficulty distribution
        if 'difficulty' in prompt_df.columns:
            diff_dist = prompt_df['difficulty'].value_counts().to_dict()
            stats['difficulty_distribution'] = diff_dist
            stats['difficulty_counts'] = dict(prompt_df['difficulty'].value_counts())
    
    # Load prompts text if available
    if prompts_json and prompts_json.exists():
        with open(prompts_json, 'r') as f:
            prompts_data = json.load(f)
        
        prompt_texts = [p.get('text', '') for p in prompts_data]
        lengths = [len(text.split()) for text in prompt_texts]
        
        stats['prompt_lengths'] = {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'median': float(np.median(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75))
        }
    
    return stats


def analyze_pairs(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze pair distributions.
    
    Args:
        features_df: Features dataframe with pair_id, y (judgment)
        
    Returns:
        Dictionary with pair statistics
    """
    stats = {}
    
    stats['total_pairs'] = len(features_df)
    
    # Judgment distribution (y should be 0 or 1, where 1 = A preferred)
    if 'y' in features_df.columns:
        y_counts = features_df['y'].value_counts().to_dict()
        stats['judgment_distribution'] = {
            'A_preferred': int(y_counts.get(1, 0) + y_counts.get(1.0, 0)),
            'B_preferred': int(y_counts.get(0, 0) + y_counts.get(0.0, 0))
        }
        stats['judgment_balance'] = float(features_df['y'].mean())
    
    # Pairs per prompt
    if 'prompt_id' in features_df.columns:
        pairs_per_prompt = features_df.groupby('prompt_id').size()
        stats['pairs_per_prompt'] = {
            'mean': float(pairs_per_prompt.mean()),
            'std': float(pairs_per_prompt.std()),
            'min': int(pairs_per_prompt.min()),
            'max': int(pairs_per_prompt.max()),
            'median': float(pairs_per_prompt.median())
        }
    
    # Split distribution
    if 'split' in features_df.columns:
        split_counts = features_df['split'].value_counts().to_dict()
        stats['split_distribution'] = split_counts
    
    # Category x Difficulty cross-tabulation
    if 'category' in features_df.columns and 'difficulty' in features_df.columns:
        crosstab = pd.crosstab(features_df['category'], features_df['difficulty'])
        stats['category_difficulty_crosstab'] = crosstab.to_dict()
    
    return stats


def analyze_features(features_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze expert feature distributions (Δz).
    
    Args:
        features_df: Features dataframe with dz_alpha, dz_beta, dz_gamma, dz_delta
        
    Returns:
        Dictionary with feature statistics
    """
    stats = {}
    
    expert_cols = ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta']
    available_cols = [col for col in expert_cols if col in features_df.columns]
    
    if not available_cols:
        return stats
    
    # Per-expert statistics
    for col in available_cols:
        expert_name = col.replace('dz_', '')
        data = features_df[col].dropna()
        
        stats[f'{expert_name}_stats'] = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'median': float(data.median()),
            'q25': float(data.quantile(0.25)),
            'q75': float(data.quantile(0.75)),
            'positive_ratio': float((data > 0).mean()),
            'negative_ratio': float((data < 0).mean()),
            'zero_ratio': float((data == 0).mean())
        }
    
    # Correlation matrix
    corr_matrix = features_df[available_cols].corr()
    stats['correlations'] = corr_matrix.to_dict()
    
    # Pairwise correlations summary
    correlations_list = []
    for i, col1 in enumerate(available_cols):
        for col2 in available_cols[i+1:]:
            corr_val = corr_matrix.loc[col1, col2]
            correlations_list.append({
                'expert1': col1.replace('dz_', ''),
                'expert2': col2.replace('dz_', ''),
                'correlation': float(corr_val)
            })
    stats['pairwise_correlations'] = correlations_list
    
    return stats


def analyze_movies(
    movie_text_path: Optional[Path] = None,
    emotion_index_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Analyze movie metadata distributions.
    
    Args:
        movie_text_path: Path to movie_text.parquet
        emotion_index_path: Path to emotion index
        
    Returns:
        Dictionary with movie statistics
    """
    stats = {}
    
    if movie_text_path and movie_text_path.exists():
        movies_df = pd.read_parquet(movie_text_path)
        
        stats['total_movies'] = len(movies_df)
        
        # Language distribution
        if 'original_language' in movies_df.columns:
            lang_counts = movies_df['original_language'].value_counts().head(10).to_dict()
            stats['top_languages'] = lang_counts
        
        # Year distribution
        if 'release_date' in movies_df.columns:
            # Extract year
            try:
                years = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year
                stats['year_range'] = {
                    'min': int(years.min()) if not years.isna().all() else None,
                    'max': int(years.max()) if not years.isna().all() else None,
                    'mean': float(years.mean()) if not years.isna().all() else None
                }
            except:
                pass
        
        # Type distribution
        if 'type' in movies_df.columns:
            type_counts = movies_df['type'].value_counts().to_dict()
            stats['type_distribution'] = type_counts
        
        # Text length statistics
        if 'overview' in movies_df.columns:
            overview_lengths = movies_df['overview'].fillna('').str.split().str.len()
            stats['overview_lengths'] = {
                'mean': float(overview_lengths.mean()),
                'median': float(overview_lengths.median()),
                'max': int(overview_lengths.max())
            }
        
        if 'plot' in movies_df.columns:
            plot_lengths = movies_df['plot'].fillna('').str.split().str.len()
            stats['plot_lengths'] = {
                'mean': float(plot_lengths.mean()),
                'median': float(plot_lengths.median()),
                'max': int(plot_lengths.max())
            }
    
    # Emotion distributions
    if emotion_index_path and emotion_index_path.exists():
        try:
            if emotion_index_path.suffix == '.json':
                with open(emotion_index_path, 'r') as f:
                    emotion_data = json.load(f)
            elif emotion_index_path.suffix == '.parquet':
                emotion_df = pd.read_parquet(emotion_index_path)
                emotion_data = emotion_df.to_dict('records')
            
            # Analyze emotion distributions
            emotions = ['Joy', 'Trust', 'Fear', 'Anticipation', 'Sadness', 'Anger', 'Surprise', 'Disgust']
            emotion_stats = {}
            
            if isinstance(emotion_data, dict):
                # Format: {movieId: {emotion: prob}}
                for emo in emotions:
                    values = [v.get(emo, 0) for v in emotion_data.values() if isinstance(v, dict)]
                    if values:
                        emotion_stats[emo] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'max': float(np.max(values))
                        }
            
            stats['emotion_distributions'] = emotion_stats
        except Exception as e:
            stats['emotion_error'] = str(e)
    
    return stats


def generate_dataset_summary(
    features_df: pd.DataFrame,
    output_dir: Path,
    prompts_json: Optional[Path] = None,
    movie_text_path: Optional[Path] = None,
    emotion_index_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Generate comprehensive dataset analysis.
    
    Args:
        features_df: Main features dataframe
        output_dir: Directory to save outputs
        prompts_json: Optional path to prompts.json
        movie_text_path: Optional path to movie_text.parquet
        emotion_index_path: Optional path to emotion index
        
    Returns:
        Complete statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'prompts': analyze_prompts(features_df, prompts_json),
        'pairs': analyze_pairs(features_df),
        'features': analyze_features(features_df),
        'movies': analyze_movies(movie_text_path, emotion_index_path)
    }
    
    # Save individual components
    for component, stats in summary.items():
        output_file = output_dir / f"{component}_statistics.json"
        with open(output_file, 'w') as f:
            json.dump(convert_to_serializable(stats), f, indent=2)
        print(f"Saved {component} statistics to {output_file}")
    
    # Save complete summary
    summary_file = output_dir / "dataset_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    print(f"Saved complete dataset summary to {summary_file}")
    
    # Generate CSV tables for easy viewing
    if 'category' in features_df.columns:
        cat_table = features_df.groupby('category').agg({
            'prompt_id': 'nunique',
            'pair_id': 'count'
        }).rename(columns={'prompt_id': 'num_prompts', 'pair_id': 'num_pairs'})
        cat_table.to_csv(output_dir / "category_table.csv")
    
    if 'difficulty' in features_df.columns:
        diff_table = features_df.groupby('difficulty').agg({
            'prompt_id': 'nunique',
            'pair_id': 'count'
        }).rename(columns={'prompt_id': 'num_prompts', 'pair_id': 'num_pairs'})
        diff_table.to_csv(output_dir / "difficulty_table.csv")
    
    # Feature correlation matrix
    expert_cols = [col for col in ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta'] 
                   if col in features_df.columns]
    if expert_cols:
        corr_df = features_df[expert_cols].corr()
        corr_df.to_csv(output_dir / "feature_correlations.csv")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument("--features", required=True, help="Path to features parquet")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--prompts_json", help="Optional path to prompts.json")
    parser.add_argument("--movie_text", help="Optional path to movie_text.parquet")
    parser.add_argument("--emotion_index", help="Optional path to emotion index")
    
    args = parser.parse_args()
    
    features_df = pd.read_parquet(args.features)
    output_dir = Path(args.output_dir)
    
    summary = generate_dataset_summary(
        features_df=features_df,
        output_dir=output_dir,
        prompts_json=Path(args.prompts_json) if args.prompts_json else None,
        movie_text_path=Path(args.movie_text) if args.movie_text else None,
        emotion_index_path=Path(args.emotion_index) if args.emotion_index else None
    )
    
    print("\n=== Dataset Summary ===")
    print(f"Total prompts: {summary['prompts'].get('total_prompts', 'N/A')}")
    print(f"Total pairs: {summary['pairs'].get('total_pairs', 'N/A')}")
    print(f"Total movies: {summary['movies'].get('total_movies', 'N/A')}")

