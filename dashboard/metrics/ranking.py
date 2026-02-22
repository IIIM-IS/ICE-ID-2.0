"""
Ranking metrics for entity resolution.

Provides Precision@K, Recall@K, MAP for evaluating ranking quality.
"""

from typing import Dict, List, Tuple
import numpy as np


def precision_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute Precision@K: fraction of top-K predictions that are correct.
    
    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        k: Number of top predictions to consider
        
    Returns:
        Precision@K value
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    
    k = min(k, len(y_scores))
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_true = y_true[top_k_indices]
    
    return float(top_k_true.sum() / k)


def recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@K: fraction of positive examples found in top-K.
    
    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        k: Number of top predictions to consider
        
    Returns:
        Recall@K value
    """
    if len(y_true) == 0 or k <= 0:
        return 0.0
    
    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0
    
    k = min(k, len(y_scores))
    top_k_indices = np.argsort(y_scores)[::-1][:k]
    top_k_true = y_true[top_k_indices]
    
    return float(top_k_true.sum() / n_positives)


def average_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute Average Precision (AP): area under precision-recall curve.
    
    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        
    Returns:
        Average Precision value
    """
    if len(y_true) == 0:
        return 0.0
    
    n_positives = y_true.sum()
    if n_positives == 0:
        return 0.0
    
    # Sort by scores descending
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Compute precision at each positive
    ap = 0.0
    n_correct = 0
    
    for i, is_correct in enumerate(y_true_sorted):
        if is_correct:
            n_correct += 1
            precision_at_i = n_correct / (i + 1)
            ap += precision_at_i
    
    return float(ap / n_positives)


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive ranking metrics.
    
    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        k_values: List of K values for P@K and R@K (default: [5, 10, 20, 50, 100])
        
    Returns:
        Dictionary containing P@K, R@K for each K, and MAP
    """
    if k_values is None:
        # Default to number of positives if available
        n_positives = int(y_true.sum()) if len(y_true) > 0 else 5
        k_values = [min(n_positives, k) for k in [5, 10, 20, 50, 100]]
    
    metrics = {}
    
    for k in k_values:
        if k > 0:
            metrics[f"p_at_{k}"] = precision_at_k(y_true, y_scores, k)
            metrics[f"r_at_{k}"] = recall_at_k(y_true, y_scores, k)
    
    metrics["map"] = average_precision(y_true, y_scores)
    
    return metrics

