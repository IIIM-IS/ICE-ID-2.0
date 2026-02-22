"""
Pairwise classification metrics for entity resolution.

Computes precision, recall, F1, accuracy, and AUC for pair-based matching tasks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def _roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
    """
    Compute ROC AUC using the Mann–Whitney U statistic.

    Args:
        y_true: Binary ground truth labels (0/1).
        y_scores: Continuous scores (higher means more likely positive).

    Returns:
        AUC in [0, 1] if both classes are present, otherwise None.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None

    order = np.argsort(y_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1, dtype=float)

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    rank_sum_pos = float(ranks[pos].sum())
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_pos / (n_pos * n_neg)
    return float(auc)


def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard pairwise classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_scores: Optional prediction scores/probabilities for AUC
        
    Returns:
        Dictionary containing precision, recall, F1, accuracy, and optionally AUC
    """
    if len(y_true) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "auc": None,
            "n_samples": 0,
        }

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / len(y_true)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "n_samples": len(y_true),
    }
    
    metrics["auc"] = _roc_auc_score(y_true, y_scores) if y_scores is not None else None
    
    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 101,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the threshold that maximizes a given metric.
    
    Args:
        y_true: Ground truth binary labels
        y_scores: Prediction scores
        metric: Metric to optimize ("f1", "precision", "recall", "accuracy")
        n_thresholds: Number of thresholds to test
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best_value = -1
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        metrics_dict = compute_pairwise_metrics(y_true, y_pred, y_scores)
        value = metrics_dict.get(metric, 0)
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
            best_metrics = metrics_dict
    
    return float(best_threshold), best_metrics

