"""
Clustering metrics for entity resolution.

Provides ARI, B-cubed, and other clustering quality metrics with sanity checks.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from collections import defaultdict


def _comb2(n: int) -> float:
    """Compute n choose 2 as a float."""
    if n < 2:
        return 0.0
    return float(n * (n - 1) / 2)


def _adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index (ARI) for two cluster labelings.

    Args:
        true_labels: Array of true cluster labels (non-negative integers).
        pred_labels: Array of predicted cluster labels (non-negative integers).

    Returns:
        ARI score in [-1, 1].
    """
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    if true_labels.size < 2:
        return 0.0

    _, true_idx = np.unique(true_labels, return_inverse=True)
    _, pred_idx = np.unique(pred_labels, return_inverse=True)

    n_true = int(true_idx.max()) + 1
    n_pred = int(pred_idx.max()) + 1

    contingency = np.zeros((n_true, n_pred), dtype=int)
    np.add.at(contingency, (true_idx, pred_idx), 1)

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


def _normalized_mutual_info(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information (NMI) using geometric mean normalization.

    Args:
        true_labels: Array of true cluster labels (non-negative integers).
        pred_labels: Array of predicted cluster labels (non-negative integers).

    Returns:
        NMI score in [0, 1].
    """
    true_labels = np.asarray(true_labels, dtype=int)
    pred_labels = np.asarray(pred_labels, dtype=int)

    if true_labels.size == 0:
        return 0.0

    _, true_idx = np.unique(true_labels, return_inverse=True)
    _, pred_idx = np.unique(pred_labels, return_inverse=True)

    n_true = int(true_idx.max()) + 1
    n_pred = int(pred_idx.max()) + 1

    contingency = np.zeros((n_true, n_pred), dtype=float)
    np.add.at(contingency, (true_idx, pred_idx), 1.0)

    n = float(contingency.sum())
    if n == 0.0:
        return 0.0

    a = contingency.sum(axis=1, keepdims=True)
    b = contingency.sum(axis=0, keepdims=True)

    pij = contingency / n
    pi = a / n
    pj = b / n

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (pij / (pi @ pj))
        mi = np.nansum(pij * np.log(np.where(pij > 0, ratio, 1.0)))

    hu = -float(np.sum(pi[pi > 0] * np.log(pi[pi > 0])))
    hv = -float(np.sum(pj[pj > 0] * np.log(pj[pj > 0])))

    denom = float(np.sqrt(hu * hv)) if hu > 0 and hv > 0 else 0.0
    if denom == 0.0:
        return 0.0

    return float(mi / denom)


def validate_clustering_inputs(
    pred_clusters: List[int],
    true_clusters: List[int],
    record_ids: Optional[List[int]] = None,
) -> Tuple[bool, List[str]]:
    """
    Perform sanity checks on clustering inputs.
    
    Args:
        pred_clusters: Predicted cluster assignments
        true_clusters: Ground truth cluster assignments  
        record_ids: Optional list of record IDs for debugging
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check length alignment
    if len(pred_clusters) != len(true_clusters):
        issues.append(
            f"Length mismatch: pred={len(pred_clusters)}, true={len(true_clusters)}"
        )
        return False, issues
    
    if len(pred_clusters) == 0:
        issues.append("Empty cluster assignments")
        return False, issues
    
    # Check for valid cluster IDs (non-negative integers or -1 for noise)
    pred_arr = np.array(pred_clusters)
    true_arr = np.array(true_clusters)
    
    if np.any(pred_arr < -1):
        issues.append(f"Invalid predicted cluster IDs (< -1): {np.unique(pred_arr[pred_arr < -1])}")
    
    if np.any(true_arr < -1):
        issues.append(f"Invalid true cluster IDs (< -1): {np.unique(true_arr[true_arr < -1])}")
    
    # Check for non-empty clusters (excluding noise label -1)
    pred_valid = pred_arr[pred_arr >= 0]
    true_valid = true_arr[true_arr >= 0]
    
    if len(pred_valid) == 0:
        issues.append("All predicted cluster IDs are noise (-1)")
    
    if len(true_valid) == 0:
        issues.append("All true cluster IDs are noise (-1)")
    
    # Check universe of nodes matches (if record_ids provided)
    if record_ids is not None:
        if len(record_ids) != len(pred_clusters):
            issues.append(
                f"Record IDs length mismatch: {len(record_ids)} vs clusters {len(pred_clusters)}"
            )
    
    return len(issues) == 0, issues


def bcubed_precision_recall_f1(
    pred_clusters: List[int],
    true_clusters: List[int],
) -> Dict[str, float]:
    """
    Compute B-cubed precision, recall, and F1.
    
    B-cubed evaluates clustering quality at the entity level rather than cluster level.
    It handles overlapping clusters and is less sensitive to cluster size distribution.
    
    Args:
        pred_clusters: Predicted cluster assignments
        true_clusters: Ground truth cluster assignments
        
    Returns:
        Dictionary with b3_precision, b3_recall, b3_f1
    """
    n = len(pred_clusters)
    if n == 0:
        return {"b3_precision": 0.0, "b3_recall": 0.0, "b3_f1": 0.0}
    
    # Build cluster membership maps
    pred_map = defaultdict(set)
    true_map = defaultdict(set)
    
    for idx, (pred_c, true_c) in enumerate(zip(pred_clusters, true_clusters)):
        if pred_c >= 0:  # Ignore noise
            pred_map[pred_c].add(idx)
        if true_c >= 0:
            true_map[true_c].add(idx)
    
    precision_sum = 0.0
    recall_sum = 0.0
    
    for idx in range(n):
        pred_c = pred_clusters[idx]
        true_c = true_clusters[idx]
        
        if pred_c == -1 or true_c == -1:
            continue
        
        # Get all items in same predicted and true clusters
        pred_same = pred_map[pred_c]
        true_same = true_map[true_c]
        
        # Precision: of items in same predicted cluster, how many are truly same?
        if len(pred_same) > 0:
            precision_sum += len(pred_same & true_same) / len(pred_same)
        
        # Recall: of items in same true cluster, how many are predicted same?
        if len(true_same) > 0:
            recall_sum += len(pred_same & true_same) / len(true_same)
    
    valid_count = sum(1 for c in true_clusters if c >= 0)
    
    if valid_count == 0:
        return {"b3_precision": 0.0, "b3_recall": 0.0, "b3_f1": 0.0}
    
    precision = precision_sum / valid_count
    recall = recall_sum / valid_count
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "b3_precision": float(precision),
        "b3_recall": float(recall),
        "b3_f1": float(f1),
    }


def compute_clustering_metrics(
    pred_clusters: List[int],
    true_clusters: List[int],
    record_ids: Optional[List[int]] = None,
    validate: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive clustering metrics including ARI, NMI, and B-cubed.
    
    Args:
        pred_clusters: Predicted cluster assignments
        true_clusters: Ground truth cluster assignments
        record_ids: Optional list of record IDs
        validate: Whether to run sanity checks
        
    Returns:
        Dictionary containing ARI, NMI, B-cubed metrics, and validation results
    """
    metrics = {}
    
    # Validation
    if validate:
        is_valid, issues = validate_clustering_inputs(pred_clusters, true_clusters, record_ids)
        metrics["validation_passed"] = is_valid
        if not is_valid:
            metrics["validation_issues"] = issues
            # Return zeros for invalid inputs
            return {
                **metrics,
                "ari": 0.0,
                "nmi": 0.0,
                "b3_precision": 0.0,
                "b3_recall": 0.0,
                "b3_f1": 0.0,
                "n_pred_clusters": 0,
                "n_true_clusters": 0,
            }
    
    # Convert to numpy arrays
    pred_arr = np.array(pred_clusters)
    true_arr = np.array(true_clusters)
    
    # Filter out noise labels (-1) for ARI computation
    valid_mask = (pred_arr >= 0) & (true_arr >= 0)
    pred_valid = pred_arr[valid_mask]
    true_valid = true_arr[valid_mask]
    
    if len(pred_valid) < 2:
        # Not enough valid data points
        metrics["ari"] = 0.0
        metrics["nmi"] = 0.0
    else:
        try:
            metrics["ari"] = _adjusted_rand_index(true_valid, pred_valid)
        except Exception as e:
            metrics["ari"] = 0.0
            metrics["ari_error"] = str(e)
        
        try:
            metrics["nmi"] = _normalized_mutual_info(true_valid, pred_valid)
        except Exception as e:
            metrics["nmi"] = 0.0
            metrics["nmi_error"] = str(e)
    
    # B-cubed metrics
    try:
        b3_metrics = bcubed_precision_recall_f1(pred_clusters, true_clusters)
        metrics.update(b3_metrics)
    except Exception as e:
        metrics["b3_error"] = str(e)
        metrics["b3_precision"] = 0.0
        metrics["b3_recall"] = 0.0
        metrics["b3_f1"] = 0.0
    
    # Cluster counts
    metrics["n_pred_clusters"] = len(set(pred_arr[pred_arr >= 0]))
    metrics["n_true_clusters"] = len(set(true_arr[true_arr >= 0]))
    metrics["n_records"] = len(pred_clusters)
    metrics["n_valid_records"] = int(valid_mask.sum())
    
    return metrics


def clusters_from_edges(
    edges: pd.DataFrame,
    all_record_ids: List[int],
) -> List[int]:
    """
    Convert edge list to cluster assignments using connected components.
    
    Args:
        edges: DataFrame with columns 'id1' and 'id2'
        all_record_ids: Complete list of record IDs
        
    Returns:
        List of cluster assignments (parallel to all_record_ids)
    """
    # Build adjacency list
    adj = defaultdict(set)
    nodes_in_edges = set()
    
    for _, row in edges.iterrows():
        id1, id2 = int(row["id1"]), int(row["id2"])
        adj[id1].add(id2)
        adj[id2].add(id1)
        nodes_in_edges.add(id1)
        nodes_in_edges.add(id2)
    
    # Find connected components via BFS
    visited = set()
    cluster_id = 0
    node_to_cluster = {}
    
    def bfs(start):
        queue = [start]
        visited.add(start)
        component = []
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return component
    
    # Assign cluster IDs
    for node in nodes_in_edges:
        if node not in visited:
            component = bfs(node)
            for n in component:
                node_to_cluster[n] = cluster_id
            cluster_id += 1
    
    # Assign -1 to nodes not in any edge (singletons)
    clusters = []
    for rid in all_record_ids:
        if rid in node_to_cluster:
            clusters.append(node_to_cluster[rid])
        else:
            clusters.append(-1)
    
    return clusters

