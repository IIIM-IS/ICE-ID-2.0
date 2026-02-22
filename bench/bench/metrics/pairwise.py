"""Pairwise classification metrics."""

from typing import Dict, List
import numpy as np


def _roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute ROC AUC using the Mann–Whitney U statistic.

    Args:
        y_true: Binary ground truth labels (0/1).
        y_scores: Continuous scores (higher means more likely positive).

    Returns:
        AUC in [0, 1]. Returns 0.0 if both classes are not present.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.0

    order = np.argsort(y_scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_scores) + 1, dtype=float)

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    rank_sum_pos = float(ranks[pos].sum())
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_pos / (n_pos * n_neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute average precision (area under precision-recall curve).

    Args:
        y_true: Binary ground truth labels (0/1).
        y_scores: Continuous scores (higher means more likely positive).

    Returns:
        Average precision in [0, 1]. Returns 0.0 if there are no positives.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(-y_scores, kind="mergesort")
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    denom = tp + fp
    precision = np.where(denom > 0, tp / denom, 0.0)

    return float(precision[y_sorted == 1].sum() / n_pos)


def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray = None,
    y_scores: np.ndarray = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pairwise classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Binary predictions (optional if scores provided).
        y_scores: Confidence scores.
        threshold: Threshold for converting scores to predictions.
        
    Returns:
        Dict with precision, recall, f1, accuracy, auc, ap.
    """
    y_true = np.asarray(y_true).astype(int)
    
    if y_pred is None and y_scores is not None:
        y_pred = (np.asarray(y_scores) >= threshold).astype(int)
    elif y_pred is None:
        raise ValueError("Either y_pred or y_scores must be provided")
    
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "n_true_positives": tp,
        "n_false_positives": fp,
        "n_false_negatives": fn,
        "n_true_negatives": tn,
    }
    
    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        metrics["auc"] = _roc_auc_score(y_true, y_scores)
        metrics["ap"] = _average_precision(y_true, y_scores)
    
    return metrics

