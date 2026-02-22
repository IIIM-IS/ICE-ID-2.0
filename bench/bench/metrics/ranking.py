"""Ranking metrics for entity resolution."""

from typing import Dict, List
import numpy as np


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Compute ranking metrics.
    
    Args:
        y_true: Ground truth labels.
        y_scores: Prediction scores.
        k_values: List of k values for P@k and R@k.
        
    Returns:
        Dict with P@k, R@k for each k.
    """
    if k_values is None:
        k_values = [10, 50, 100]
    
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_labels = y_true[sorted_indices]
    
    total_positives = np.sum(y_true == 1)
    
    metrics = {}
    
    for k in k_values:
        if k > len(sorted_labels):
            k_actual = len(sorted_labels)
        else:
            k_actual = k
        
        top_k_labels = sorted_labels[:k_actual]
        true_positives_at_k = np.sum(top_k_labels == 1)
        
        p_at_k = true_positives_at_k / k_actual if k_actual > 0 else 0
        r_at_k = true_positives_at_k / total_positives if total_positives > 0 else 0
        
        metrics[f"p_at_{k}"] = p_at_k
        metrics[f"r_at_{k}"] = r_at_k
    
    return metrics

