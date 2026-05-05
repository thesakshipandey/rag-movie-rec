"""
Evaluation metrics for router performance.

Implements agreement metrics for pairwise comparisons.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def agreement_no_ties(scores: np.ndarray, y_prime: np.ndarray) -> float:
    """
    Agreement metric treating all predictions as hard decisions.
    
    Ties (score=0) are counted as wrong.
    
    Args:
        scores: [N] predicted scores (can be positive or negative)
        y_prime: [N] true labels in {-1, +1}
    
    Returns:
        Agreement fraction in [0, 1]
    """
    # Sign of score should match y_prime
    pred_sign = np.sign(scores)
    
    # Ties (pred_sign=0) are wrong
    correct = (pred_sign == y_prime)
    
    return float(np.mean(correct))


def agreement_ties_half(
    scores: np.ndarray,
    y_prime: np.ndarray,
    tol: float = 0.05,
) -> float:
    """
    Agreement metric giving partial credit to ties.
    
    If |score| <= tol, treat as tie and give 0.5 credit.
    Otherwise, give 1.0 if sign matches, 0.0 otherwise.
    
    Args:
        scores: [N] predicted scores
        y_prime: [N] true labels in {-1, +1}
        tol: Threshold for considering a score as a tie
    
    Returns:
        Agreement score in [0, 1]
    """
    credits = np.zeros_like(scores, dtype=float)
    
    # Ties: |score| <= tol → 0.5 credit
    is_tie = np.abs(scores) <= tol
    credits[is_tie] = 0.5
    
    # Non-ties: check if sign matches
    is_correct = (np.sign(scores) == y_prime)
    credits[~is_tie & is_correct] = 1.0
    credits[~is_tie & ~is_correct] = 0.0
    
    return float(np.mean(credits))


def compute_metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    score_col: str = 'score',
    y_prime_col: str = 'y_prime',
    tol: float = 0.05,
) -> pd.DataFrame:
    """
    Compute agreement metrics grouped by a column.
    
    Args:
        df: DataFrame with scores and labels
        group_col: Column to group by (e.g., 'difficulty', 'category')
        score_col: Column with predicted scores
        y_prime_col: Column with true labels {-1, +1}
        tol: Tolerance for tie detection
    
    Returns:
        DataFrame with metrics per group
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame")
    
    results = []
    
    for group_val, group_df in df.groupby(group_col):
        scores = group_df[score_col].values
        y_prime = group_df[y_prime_col].values
        
        metrics = {
            group_col: group_val,
            'n_pairs': len(scores),
            'agree_no_ties': agreement_no_ties(scores, y_prime),
            'agree_ties_0p5': agreement_ties_half(scores, y_prime, tol=tol),
        }
        results.append(metrics)
    
    return pd.DataFrame(results)


def compute_overall_metrics(
    scores: np.ndarray,
    y_prime: np.ndarray,
    tol: float = 0.05,
) -> Dict[str, float]:
    """
    Compute overall metrics.
    
    Args:
        scores: [N] predicted scores
        y_prime: [N] true labels {-1, +1}
        tol: Tolerance for tie detection
    
    Returns:
        Dictionary with metrics
    """
    return {
        'n_pairs': len(scores),
        'agree_no_ties': agreement_no_ties(scores, y_prime),
        'agree_ties_0p5': agreement_ties_half(scores, y_prime, tol=tol),
        'mean_abs_score': float(np.mean(np.abs(scores))),
        'std_score': float(np.std(scores)),
    }

