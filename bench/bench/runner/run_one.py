"""Run a single experiment."""

import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yaml

from ..core.types import DatasetSplit, EvaluationResult
from ..core.registry import get_registry
from ..core.random import set_seed
from ..metrics import (
    compute_pairwise_metrics,
    compute_clustering_metrics,
    compute_ranking_metrics,
    run_sanity_checks,
)


def run_experiment(
    config: Dict[str, Any],
    output_dir: str = None,
    verbose: bool = True
) -> EvaluationResult:
    """
    Run a single experiment from configuration.
    
    Args:
        config: Experiment configuration dict.
        output_dir: Directory to save results.
        verbose: Print progress.
        
    Returns:
        EvaluationResult with all metrics.
    """
    set_seed(config.get("seed", 42))
    
    start_time = time.time()
    
    if verbose:
        print(f"Running experiment: {config.get('name', 'unnamed')}")
    
    dataset_config = config.get("dataset", {})
    dataset_name = dataset_config.get("name", "iceid")
    dataset_cls = get_registry("datasets").get(dataset_name)
    
    dataset = dataset_cls(**{k: v for k, v in dataset_config.items() if k != "name"})
    split = dataset.load()
    
    if verbose:
        print(f"  Loaded dataset: {dataset.name}")
        print(f"  Summary: {dataset.summary()}")
    
    blocking_config = config.get("blocking", {})
    blocker_name = blocking_config.get("name", "trivial_allpairs")
    
    if blocker_name == "trivial_allpairs":
        from ..blocking.base import TrivialAllPairsBlocker
        blocker = TrivialAllPairsBlocker()
    else:
        blocker_cls = get_registry("blockers").get(blocker_name)
        blocker = blocker_cls(**blocking_config.get("params", {}))
    
    candidates = blocker.block(split)
    
    if verbose:
        print(f"  Generated {len(candidates.pairs)} candidate pairs")
    
    pairs_config = config.get("pairs", {})
    max_pairs = pairs_config.get("cap", 10000)
    
    if len(candidates.pairs) > max_pairs:
        rng = np.random.default_rng(config.get("seed", 42))
        indices = rng.choice(len(candidates.pairs), size=max_pairs, replace=False)
        pairs = [candidates.pairs[i] for i in indices]
    else:
        pairs = candidates.pairs
    
    labels = dataset.get_pair_labels(pairs)
    
    split_ratio = pairs_config.get("train_ratio", 0.7)
    n_train = int(len(pairs) * split_ratio)
    
    train_pairs = pd.DataFrame({
        "id1": [p[0] for p in pairs[:n_train]],
        "id2": [p[1] for p in pairs[:n_train]],
        "label": labels[:n_train],
    })
    test_pairs = pairs[n_train:]
    test_labels = labels[n_train:]
    
    if verbose:
        print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    
    model_config = config.get("model", {})
    model_name = model_config.get("name", "nars")
    model_cls = get_registry("models").get(model_name)
    model = model_cls(**model_config.get("params", {}))
    
    if verbose:
        print(f"  Training model: {model.name}")
    
    model.fit(split, train_pairs)
    
    if verbose:
        print(f"  Scoring test pairs...")
    
    scores = model.score(split, test_pairs)
    
    calib_config = config.get("calibration", {})
    calib_name = calib_config.get("name", "fixed_threshold")
    calib_cls = get_registry("calibrators").get(calib_name)
    calibrator = calib_cls(**calib_config.get("params", {}))
    
    calibrator.fit(scores, np.array(test_labels))
    calibrated_scores = calibrator.calibrate(scores)
    predictions = calibrator.predict(scores)
    
    pairwise = compute_pairwise_metrics(
        np.array(test_labels),
        predictions,
        calibrated_scores,
        threshold=calibrator.threshold
    )
    
    ranking = compute_ranking_metrics(
        np.array(test_labels),
        scores,
        k_values=[10, 50, 100]
    )
    
    cluster_config = config.get("clustering", {})
    clusterer_name = cluster_config.get("name", "connected_components")
    clusterer_cls = get_registry("clusterers").get(clusterer_name)
    clusterer = clusterer_cls(**cluster_config.get("params", {}))
    
    positive_edges = [
        (test_pairs[i][0], test_pairs[i][1], float(scores[i]))
        for i in range(len(test_pairs))
        if predictions[i] == 1
    ]
    
    all_ids = list(set(p[0] for p in test_pairs) | set(p[1] for p in test_pairs))
    cluster_result = clusterer.cluster(positive_edges, all_ids)
    
    ground_truth_pairs, ground_truth_labels = dataset.get_ground_truth()
    
    clustering = {}
    if ground_truth_labels:
        clustering = compute_clustering_metrics(ground_truth_labels, cluster_result.clusters)
    
    sanity = run_sanity_checks(
        y_true=np.array(test_labels),
        y_pred=predictions,
        y_scores=scores,
        true_labels=ground_truth_labels,
        predicted_clusters=cluster_result.clusters,
        pairs=test_pairs,
    )
    
    elapsed = time.time() - start_time
    
    result = EvaluationResult(
        pairwise=pairwise,
        clustering=clustering,
        ranking=ranking,
        sanity_checks=sanity,
        metadata={
            "config": config,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
            "threshold": calibrator.threshold,
        }
    )
    
    if verbose:
        print(f"\nResults:")
        print(f"  Pairwise F1: {pairwise['f1']:.4f}")
        print(f"  Precision: {pairwise['precision']:.4f}, Recall: {pairwise['recall']:.4f}")
        if 'auc' in pairwise:
            print(f"  AUC: {pairwise['auc']:.4f}")
        if 'ari' in clustering:
            print(f"  Clustering ARI: {clustering['ari']:.4f}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Sanity checks passed: {sanity.get('all_passed', False)}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    
    return result

