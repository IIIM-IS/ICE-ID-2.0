"""Display helpers for the real ICE-ID dashboard."""
import os
from typing import Dict, List, Optional, Tuple
import streamlit as st
ICEID_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _metric_explanations() -> Dict[str, str]:
    """Provides short definitions for each metric key shown in the UI.

    Returns:
        Dict[str, str]: A dictionary mapping metric names to their explanations.
    """
    return {
        "prec_labeled": "Agreement precision against available reference IDs; not independently established historical accuracy",
        "rec_labeled": "Recall on labeled pairs (true positives / (true positives + false negatives))",
        "f1_labeled": "F1 score on labeled pairs (harmonic mean of precision and recall)",
        "prec_labeled_wilson95": "Not estimated under shared-record dependence and unresolved reference truth",
        "edges_total": "Total number of predicted edges/pairs",
        "semi_novel_edges": "Edges where one node is labeled, one is unlabeled",
        "novel_edges": "Edges where both nodes are unlabeled",
        "unknown_identity_edges": "Edges touching at least one record without a reference ID",
        "est_true_new_edges": "Not extrapolated from selected labelled records to unknown identities",
        "nodes_with_any_match": "Number of nodes that have at least one predicted match",
        "cluster_stats": "Statistics about the clustering results (purity, number of clusters, etc.)",
    }

def _train_param_explanations() -> Dict[str, str]:
    """Provides short definitions for training parameters shown in the UI.

    Returns:
        Dict[str, str]: A dictionary mapping parameter names to their explanations.
    """
    return {
        "people_csv": "Path to the people.csv file containing individual records",
        "labels_csv": "Path to the labels.csv file containing ground truth cluster labels",
        "out_dir": "Output directory where results will be saved",
        "backends": "Comma-separated list of backends to run (e.g., 'gbdt,logreg')",
        "num_shards": "Number of shards to split the data into for parallel processing",
        "shard_id": "ID of the current shard (0-based)",
        "n_workers": "Number of worker processes for parallel processing",
        "max_block_size": "Maximum size of a blocking group",
        "max_pairs_per_block": "Maximum number of pairs to generate per block",
        "gbdt_neg_ratio": "Ratio of negative to positive samples for GBDT training",
        "thresh_grid": "Number of threshold values to test for optimal threshold selection",
        "seed": "Random seed for reproducibility",
        "sample_frac": "Fraction of data to use (1.0 = all data, 0.1 = 10%)",
        "fn_prefix": "Length of first name prefix for blocking",
        "pat_prefix": "Length of patronymic prefix for blocking",
        "year_bucket_width": "Width of year buckets for blocking",
        "dual_blocking": "Whether to use dual blocking (both forward and reverse)",
        "soft_filter_max_year_diff": "Maximum year difference for soft filtering",
        "soft_filter_sex": "Whether to apply sex-based soft filtering",
        "preview_window": "Fraction of data to use for preview/validation",
        "preview_limit": "Maximum number of samples for preview",
    }

def _bytes_to_human(n: int) -> str:
    """Converts bytes to human-readable format.

    Args:
        n (int): Number of bytes.

    Returns:
        str: Human-readable string (e.g., "1.2 MB").
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def _grade(value: Optional[float], bands: List[Tuple[float, str]]) -> str:
    """Grades a numerical value based on threshold bands.

    Args:
        value (Optional[float]): The numerical value to grade. Can be None.
        bands (List[Tuple[float, str]]): A list of (threshold, label) tuples, sorted in descending order of threshold.

    Returns:
        str: The corresponding label ('poor' if below all thresholds, 'unknown' if value is None).
    """
    if value is None:
        return "unknown"
    for th, lab in bands:
        if value >= th:
            return lab
    return "poor"

def _fmt_pct(x: Optional[float]) -> str:
    """Formats a float as a percentage string.

    Args:
        x (Optional[float]): The float to format (e.g., 0.951). Returns '—' if None.

    Returns:
        str: The formatted percentage string (e.g., "95.1%").
    """
    return "—" if x is None else f"{100.0 * x:.1f}%"

def _auto_conclusions(r: Dict) -> Dict[str, str]:
    """Describe observed reference agreement without claiming identity truth."""
    precision = r.get("prec_labeled")
    unknown = (r.get("semi_novel_edges") or 0) + (r.get("novel_edges") or 0)
    clusters = r.get("cluster_stats") or {}
    return {
        "prec_labeled": ("No evaluable reference-labelled edges." if precision is None else
                         f"Reference precision {_fmt_pct(precision)}. Source adjudication is needed to estimate historical identity accuracy."),
        "edges_total": f"{r.get('edges_total', 0)} predicted edges; volume alone does not establish quality.",
        "unknown_identity_edges": f"{unknown} edges touch unknown identities. No true-link count is extrapolated from the selected labelled subset.",
        "cluster_stats": f"{clusters.get('n_clusters', 'Unknown')} predicted clusters. Purity, when available, concerns reference agreement only.",
    }

def _get_available_run_directories() -> List[str]:
    """Scans the runs directory to find available run directories.

    Returns:
        List[str]: A list of available run directory paths, sorted by modification time (newest first).
    """
    runs_dir = os.path.join(ICEID_ROOT, "runs")
    from dashboard.release_paths import default_run_directory
    selected=default_run_directory(ICEID_ROOT)
    if not os.path.exists(runs_dir):
        return [selected] if os.path.isdir(selected) else []

    run_dirs = []
    for item in os.listdir(runs_dir):
        item_path = os.path.join(runs_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains any backend subdirectories or is a shard directory
            has_backends = False
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    # Check if it's a backend directory (has edges.csv or clusters.csv)
                    if (os.path.exists(os.path.join(subitem_path, "edges.csv")) or
                        os.path.exists(os.path.join(subitem_path, "clusters.csv"))):
                        has_backends = True
                        break
                    # Or if it's a shard directory
                    if subitem.startswith("shard_"):
                        has_backends = True
                        break

            if has_backends:
                run_dirs.append(item_path)

    # Sort by modification time (newest first)
    run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    if os.path.isdir(selected):
        run_dirs=[selected]+[p for p in run_dirs if os.path.abspath(p)!=os.path.abspath(selected)]
    return run_dirs

def _get_available_backends(run_dir: str) -> List[str]:
    """Scans a run directory to find available backend subdirectories.

    Args:
        run_dir (str): Path to the run directory to scan.

    Returns:
        List[str]: A list of available backend names, sorted alphabetically.
    """
    if not os.path.exists(run_dir):
        return []

    backends = []
    for item in os.listdir(run_dir):
        item_path = os.path.join(run_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a backend directory (has edges.csv or clusters.csv)
            if (os.path.exists(os.path.join(item_path, "edges.csv")) or
                os.path.exists(os.path.join(item_path, "clusters.csv"))):
                backends.append(item)

    backends.sort()
    return backends

