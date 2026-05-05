#!/usr/bin/env python
"""Error analysis module for failure pattern identification.

Analyze:
- Pairs where MoE fails but individual experts succeed
- Queries where router assigns suboptimal weights
- Failure patterns by category and difficulty
- Generate case studies with examples
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


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


def load_predictions(
    model_name: str,
    features_df: pd.DataFrame,
    weights_path: Optional[Path] = None
) -> pd.DataFrame:
    """Load model predictions for pairs.
    
    Args:
        model_name: Name of model
        features_df: Features dataframe
        weights_path: Path to model weights
        
    Returns:
        DataFrame with predictions added
    """
    # This would need to actually run the model
    # For now, we'll work with what's available in features
    return features_df.copy()


def identify_failures(
    predictions: pd.DataFrame,
    y_col: str = 'y',
    pred_col: str = 'prediction',
    tie_tol: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identify correct, incorrect, and tie predictions.
    
    Args:
        predictions: DataFrame with true labels and predictions
        y_col: Column name for true labels
        pred_col: Column name for predictions (margin scores)
        tie_tol: Tolerance for ties
        
    Returns:
        (correct_df, incorrect_df, ties_df)
    """
    if pred_col not in predictions.columns:
        # Compute from dz columns if available
        if 'dz_alpha' in predictions.columns:
            # Use simple sum as proxy
            expert_cols = [c for c in ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta'] 
                          if c in predictions.columns]
            predictions[pred_col] = predictions[expert_cols].sum(axis=1) / len(expert_cols)
    
    df = predictions.copy()
    
    # Determine agreement
    y_judgment = np.where(df[y_col] > 0.5, 1, -1)
    pred_sign = np.where(df[pred_col].abs() <= tie_tol, 0, np.sign(df[pred_col]))
    
    df['agreement'] = pred_sign * y_judgment
    
    correct_df = df[df['agreement'] == 1].copy()
    incorrect_df = df[df['agreement'] == -1].copy()
    ties_df = df[df['agreement'] == 0].copy()
    
    return correct_df, incorrect_df, ties_df


def analyze_expert_disagreements(
    features_df: pd.DataFrame,
    expert_cols: List[str] = None
) -> pd.DataFrame:
    """Analyze cases where experts disagree.
    
    Args:
        features_df: Features with expert Δz scores
        expert_cols: List of expert feature columns
        
    Returns:
        DataFrame with disagreement analysis
    """
    if expert_cols is None:
        expert_cols = [c for c in ['dz_alpha', 'dz_beta', 'dz_gamma', 'dz_delta'] 
                      if c in features_df.columns]
    
    if not expert_cols:
        return pd.DataFrame()
    
    df = features_df.copy()
    
    # Sign of each expert
    for col in expert_cols:
        df[f'{col}_sign'] = np.sign(df[col])
    
    sign_cols = [f'{col}_sign' for col in expert_cols]
    
    # Agreement: all experts agree on sign
    df['all_agree'] = df[sign_cols].nunique(axis=1) == 1
    
    # Count of positive vs negative
    df['num_positive'] = (df[sign_cols] == 1).sum(axis=1)
    df['num_negative'] = (df[sign_cols] == -1).sum(axis=1)
    df['num_zero'] = (df[sign_cols] == 0).sum(axis=1)
    
    # Disagreement cases
    disagreement_df = df[~df['all_agree']].copy()
    
    return disagreement_df


def identify_moe_specific_failures(
    features_df: pd.DataFrame,
    moe_predictions: np.ndarray,
    expert_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    tie_tol: float = 0.05
) -> pd.DataFrame:
    """Find cases where MoE fails but at least one expert succeeds.
    
    Args:
        features_df: Features dataframe
        moe_predictions: MoE margin predictions
        expert_predictions: Dict of expert_name -> predictions
        y_true: True labels
        tie_tol: Tolerance for ties
        
    Returns:
        DataFrame of interesting failure cases
    """
    df = features_df.copy()
    df['y_true'] = y_true
    df['moe_pred'] = moe_predictions
    
    # MoE agreement
    y_judgment = np.where(y_true > 0.5, 1, -1)
    moe_sign = np.where(np.abs(moe_predictions) <= tie_tol, 0, np.sign(moe_predictions))
    df['moe_agreement'] = moe_sign * y_judgment
    
    # Expert agreements
    for expert_name, expert_pred in expert_predictions.items():
        expert_sign = np.where(np.abs(expert_pred) <= tie_tol, 0, np.sign(expert_pred))
        df[f'{expert_name}_agreement'] = expert_sign * y_judgment
    
    # MoE fails (incorrect)
    moe_failures = df[df['moe_agreement'] == -1].copy()
    
    # At least one expert succeeds
    expert_agreement_cols = [f'{name}_agreement' for name in expert_predictions.keys()]
    moe_failures['any_expert_correct'] = (moe_failures[expert_agreement_cols] == 1).any(axis=1)
    
    interesting_failures = moe_failures[moe_failures['any_expert_correct']].copy()
    
    return interesting_failures


def extract_failure_patterns(
    failures_df: pd.DataFrame,
    group_by: List[str] = None
) -> pd.DataFrame:
    """Extract patterns from failure cases.
    
    Args:
        failures_df: DataFrame of failure cases
        group_by: Columns to group by (e.g., category, difficulty)
        
    Returns:
        Summary DataFrame
    """
    if group_by is None:
        group_by = []
        if 'category' in failures_df.columns:
            group_by.append('category')
        if 'difficulty' in failures_df.columns:
            group_by.append('difficulty')
    
    if not group_by:
        return pd.DataFrame({'total_failures': [len(failures_df)]})
    
    summary = failures_df.groupby(group_by, dropna=False).agg({
        'pair_id': 'count',
        'prompt_id': 'nunique'
    }).rename(columns={
        'pair_id': 'num_failures',
        'prompt_id': 'num_prompts'
    })
    
    return summary


def generate_case_studies(
    failures_df: pd.DataFrame,
    prompts_data: Optional[Dict] = None,
    n_cases: int = 10
) -> List[Dict[str, Any]]:
    """Generate detailed case studies from failure examples.
    
    Args:
        failures_df: DataFrame of failures
        prompts_data: Optional prompt text data
        n_cases: Number of cases to generate
        
    Returns:
        List of case study dictionaries
    """
    case_studies = []
    
    # Select diverse cases (by category/difficulty if available)
    if 'category' in failures_df.columns:
        # Sample from each category
        samples = []
        for cat in failures_df['category'].unique():
            cat_failures = failures_df[failures_df['category'] == cat]
            n_samples = min(2, len(cat_failures))
            samples.append(cat_failures.sample(n=n_samples, random_state=42))
        sample_df = pd.concat(samples) if samples else failures_df.head(n_cases)
    else:
        sample_df = failures_df.head(n_cases)
    
    for idx, row in sample_df.iterrows():
        case = {
            'pair_id': row.get('pair_id', idx),
            'prompt_id': row.get('prompt_id', None),
            'category': row.get('category', 'unknown'),
            'difficulty': row.get('difficulty', 'unknown'),
            'true_preference': 'A' if row.get('y', 0.5) > 0.5 else 'B',
        }
        
        # Add expert scores
        for expert in ['alpha', 'beta', 'gamma', 'delta']:
            col = f'dz_{expert}'
            if col in row:
                case[f'{expert}_score'] = float(row[col])
        
        # Add agreement info
        for col in row.index:
            if 'agreement' in col:
                case[col] = int(row[col])
        
        case_studies.append(case)
    
    return case_studies


def generate_error_analysis(
    features_df: pd.DataFrame,
    output_dir: Path,
    split: str = 'test',
    prompts_data: Optional[Dict] = None
) -> Dict[str, Any]:
    """Comprehensive error analysis.
    
    Args:
        features_df: Features dataframe with predictions
        output_dir: Where to save outputs
        split: Which split to analyze
        prompts_data: Optional prompt text data
        
    Returns:
        Error analysis summary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'split' in features_df.columns:
        df = features_df[features_df['split'] == split].copy()
    else:
        df = features_df.copy()
    
    print(f"Analyzing {len(df)} pairs from {split} split...")
    
    analysis = {}
    
    # Expert disagreement analysis
    print("Analyzing expert disagreements...")
    disagreements = analyze_expert_disagreements(df)
    analysis['disagreement_rate'] = len(disagreements) / len(df) if len(df) > 0 else 0
    analysis['num_disagreements'] = len(disagreements)
    
    if not disagreements.empty:
        disagreements.to_csv(output_dir / "expert_disagreements.csv", index=False)
        print(f"Saved {len(disagreements)} disagreement cases")
    
    # Failure patterns by category
    if 'category' in df.columns:
        print("Extracting failure patterns by category...")
        patterns = extract_failure_patterns(df, group_by=['category'])
        patterns.to_csv(output_dir / "failure_patterns_by_category.csv")
        analysis['patterns_by_category'] = patterns.to_dict()
    
    # Failure patterns by difficulty
    if 'difficulty' in df.columns:
        print("Extracting failure patterns by difficulty...")
        patterns = extract_failure_patterns(df, group_by=['difficulty'])
        patterns.to_csv(output_dir / "failure_patterns_by_difficulty.csv")
        analysis['patterns_by_difficulty'] = patterns.to_dict()
    
    # Case studies
    print("Generating case studies...")
    if not disagreements.empty:
        case_studies = generate_case_studies(disagreements, prompts_data, n_cases=20)
        analysis['case_studies'] = case_studies
        
        with open(output_dir / "case_studies.json", 'w') as f:
            json.dump(convert_to_serializable(case_studies), f, indent=2)
        print(f"Generated {len(case_studies)} case studies")
    
    # Save complete analysis
    with open(output_dir / "error_analysis_summary.json", 'w') as f:
        json.dump(convert_to_serializable(analysis), f, indent=2)
    print(f"Saved error analysis to {output_dir / 'error_analysis_summary.json'}")
    
    return analysis


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Error analysis")
    parser.add_argument("--features", required=True, help="Path to features parquet")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split", default="test", help="Which split to analyze")
    parser.add_argument("--prompts_json", help="Optional path to prompts.json")
    
    args = parser.parse_args()
    
    features_df = pd.read_parquet(args.features)
    output_dir = Path(args.output_dir)
    
    prompts_data = None
    if args.prompts_json:
        with open(args.prompts_json, 'r') as f:
            prompts_data = json.load(f)
    
    analysis = generate_error_analysis(
        features_df=features_df,
        output_dir=output_dir,
        split=args.split,
        prompts_data=prompts_data
    )
    
    print("\n=== Error Analysis Complete ===")
    print(f"Disagreement rate: {analysis.get('disagreement_rate', 0):.2%}")
    print(f"Number of case studies: {len(analysis.get('case_studies', []))}")

