"""Sanity checks for evaluation validity."""

from typing import Dict, List, Set
import numpy as np

from ..core.types import Pair, ClusterResult


def run_sanity_checks(
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    y_scores: np.ndarray = None,
    true_labels: Dict[int, int] = None,
    predicted_clusters: List[List[int]] = None,
    pairs: List[Pair] = None,
) -> Dict[str, bool]:
    """
    Run sanity checks on evaluation inputs.
    
    Catches common issues like:
    - All predictions are same class
    - Labels and predictions have different universes
    - Empty clusters
    - Score symmetry violations
    
    Args:
        y_true: Ground truth pair labels.
        y_pred: Predicted pair labels.
        y_scores: Prediction scores.
        true_labels: Ground truth cluster labels.
        predicted_clusters: Predicted clusters.
        pairs: Pairs that were evaluated.
        
    Returns:
        Dict of check names to pass/fail booleans.
    """
    checks = {}
    
    if y_true is not None:
        y_true = np.asarray(y_true)
        checks["has_positive_labels"] = np.sum(y_true == 1) > 0
        checks["has_negative_labels"] = np.sum(y_true == 0) > 0
        checks["labels_not_empty"] = len(y_true) > 0
    
    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        checks["has_positive_preds"] = np.sum(y_pred == 1) > 0
        checks["has_negative_preds"] = np.sum(y_pred == 0) > 0
        checks["preds_not_all_same"] = len(np.unique(y_pred)) > 1
    
    if y_scores is not None:
        y_scores = np.asarray(y_scores)
        checks["scores_not_constant"] = np.std(y_scores) > 1e-10
        checks["scores_in_valid_range"] = (
            np.min(y_scores) >= 0 and np.max(y_scores) <= 1.01
        ) or (
            np.min(y_scores) >= -100 and np.max(y_scores) <= 100
        )
    
    if true_labels is not None:
        unique_true = set(true_labels.values())
        checks["true_clusters_not_empty"] = len(unique_true) > 0
        checks["true_multiple_clusters"] = len(unique_true) > 1
    
    if predicted_clusters is not None:
        checks["pred_clusters_not_empty"] = len(predicted_clusters) > 0
        checks["pred_no_empty_clusters"] = all(
            len(c) > 0 for c in predicted_clusters
        )
        
        all_members = []
        for c in predicted_clusters:
            all_members.extend(c)
        checks["pred_no_duplicate_members"] = (
            len(all_members) == len(set(all_members))
        )
    
    if true_labels is not None and predicted_clusters is not None:
        true_ids = set(true_labels.keys())
        pred_ids = set()
        for c in predicted_clusters:
            pred_ids.update(c)
        
        overlap = true_ids & pred_ids
        checks["label_universe_overlap"] = len(overlap) > 0
        checks["label_universe_match_ratio"] = (
            len(overlap) / max(len(true_ids), len(pred_ids), 1) > 0.5
        )
    
    if pairs is not None:
        pair_set = set()
        for id1, id2 in pairs:
            if (id2, id1) in pair_set:
                checks["pair_symmetry_consistent"] = False
                break
            pair_set.add((id1, id2))
        else:
            checks["pair_symmetry_consistent"] = True
    
    checks["all_passed"] = all(checks.values())
    
    return checks

