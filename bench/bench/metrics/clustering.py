"""Clustering metrics including ARI and B³."""

from typing import Dict, List
from collections import defaultdict
import numpy as np


def _comb2(n: int) -> float:
    """Compute n choose 2 as a float."""
    if n < 2:
        return 0.0
    return float(n * (n - 1) / 2)


def _adjusted_rand_index(true_arr: List[int], pred_arr: List[int]) -> float:
    """
    Compute Adjusted Rand Index (ARI) for two labelings.

    Args:
        true_arr: True cluster labels aligned to records.
        pred_arr: Predicted cluster labels aligned to records.

    Returns:
        ARI score in [-1, 1].
    """
    true_labels = np.asarray(true_arr, dtype=int)
    pred_labels = np.asarray(pred_arr, dtype=int)

    if true_labels.size < 2:
        return 0.0

    _, t_idx = np.unique(true_labels, return_inverse=True)
    _, p_idx = np.unique(pred_labels, return_inverse=True)

    n_true = int(t_idx.max()) + 1
    n_pred = int(p_idx.max()) + 1

    contingency = np.zeros((n_true, n_pred), dtype=int)
    np.add.at(contingency, (t_idx, p_idx), 1)

    sum_comb = float(np.sum([_comb2(int(x)) for x in contingency.ravel()]))
    sum_true = float(np.sum([_comb2(int(x)) for x in contingency.sum(axis=1)]))
    sum_pred = float(np.sum([_comb2(int(x)) for x in contingency.sum(axis=0)]))

    n = int(contingency.sum())
    total = _comb2(n)
    if total == 0.0:
        return 0.0

    expected = (sum_true * sum_pred) / total
    max_index = 0.5 * (sum_true + sum_pred)
    denom = max_index - expected
    if denom == 0.0:
        return 0.0

    return float((sum_comb - expected) / denom)


def compute_clustering_metrics(
    true_labels: Dict[int, int],
    predicted_clusters: List[List[int]]
) -> Dict[str, float]:
    """
    Compute clustering metrics.
    
    Args:
        true_labels: Ground truth mapping {record_id: cluster_id}.
        predicted_clusters: List of predicted clusters.
        
    Returns:
        Dict with ARI, B³ precision/recall/F1.
    """
    pred_labels = {}
    for cluster_id, members in enumerate(predicted_clusters):
        for record_id in members:
            pred_labels[record_id] = cluster_id
    
    common_ids = sorted(set(true_labels.keys()) & set(pred_labels.keys()))
    
    if len(common_ids) < 2:
        return {
            "ari": 0.0,
            "b3_precision": 0.0,
            "b3_recall": 0.0,
            "b3_f1": 0.0,
            "n_common_records": len(common_ids),
        }
    
    true_arr = [true_labels[id_] for id_ in common_ids]
    pred_arr = [pred_labels[id_] for id_ in common_ids]
    
    ari = _adjusted_rand_index(true_arr, pred_arr)
    
    b3_p, b3_r, b3_f1 = compute_b3_metrics(true_labels, pred_labels, common_ids)
    
    return {
        "ari": ari,
        "b3_precision": b3_p,
        "b3_recall": b3_r,
        "b3_f1": b3_f1,
        "n_common_records": len(common_ids),
        "n_true_clusters": len(set(true_arr)),
        "n_pred_clusters": len(set(pred_arr)),
    }


def compute_b3_metrics(
    true_labels: Dict[int, int],
    pred_labels: Dict[int, int],
    record_ids: List[int]
) -> tuple:
    """
    Compute B-cubed precision, recall, and F1.
    
    B³ is preferred over ARI for skewed cluster sizes.
    
    Args:
        true_labels: Ground truth cluster labels.
        pred_labels: Predicted cluster labels.
        record_ids: Records to evaluate.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    true_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    
    for record_id in record_ids:
        true_clusters[true_labels[record_id]].add(record_id)
        pred_clusters[pred_labels[record_id]].add(record_id)
    
    precisions = []
    recalls = []
    
    for record_id in record_ids:
        true_cluster = true_clusters[true_labels[record_id]]
        pred_cluster = pred_clusters[pred_labels[record_id]]
        
        intersection = len(true_cluster & pred_cluster)
        
        precision = intersection / len(pred_cluster) if pred_cluster else 0
        recall = intersection / len(true_cluster) if true_cluster else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0
    
    return avg_precision, avg_recall, f1

