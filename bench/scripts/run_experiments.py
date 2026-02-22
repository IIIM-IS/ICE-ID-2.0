#!/usr/bin/env python3
"""
Consolidated experiment runner for the ICE-ID benchmark.

Usage:
    python scripts/run_experiments.py nars-rerun       # Run NARS on all datasets
    python scripts/run_experiments.py nars-graph      # Run NARS graph evaluation (ranking/clustering)
    python scripts/run_experiments.py nars-calibration # Run NARS calibration sensitivity
    python scripts/run_experiments.py all             # Run everything
"""
import os
import sys
import json
import time
import resource
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

# Paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts"
RAW_DATA = BASE_DIR.parent / "raw_data"
DEEPMATCHER_DATA = BASE_DIR / "deepmatcher_data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def get_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _comb2(n: int) -> float:
    """Compute n choose 2 as a float."""
    if n < 2:
        return 0.0
    return float(n * (n - 1) / 2)


def _adjusted_rand_index(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Compute Adjusted Rand Index (ARI) for two labelings.

    Args:
        true_labels: True cluster labels aligned to records.
        pred_labels: Predicted cluster labels aligned to records.

    Returns:
        ARI score in [-1, 1].
    """
    t = np.asarray(true_labels, dtype=int)
    p = np.asarray(pred_labels, dtype=int)

    if t.size < 2:
        return 0.0

    _, t_idx = np.unique(t, return_inverse=True)
    _, p_idx = np.unique(p, return_inverse=True)

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


# =============================================================================
# NARS RERUN ALL DATASETS
# =============================================================================

def run_nars_rerun(max_train_pairs: int = 2000, max_test_pairs: int = 2000):
    """Run NARS (OpenNARS-for-Applications) on ICE-ID + all DeepMatcher datasets with pairwise evaluation.

    Args:
        max_train_pairs: Maximum number of labeled pairs to use for training per dataset.
        max_test_pairs: Maximum number of labeled pairs to use for evaluation per dataset.
    """
    from bench.data.iceid import IceIdDataset
    from bench.data.deepmatcher import DeepMatcherDataset
    from bench.blocking.token_blocking import TokenBlocker
    from bench.models.opennars_adapter import OpenNARSModel
    from bench.metrics.pairwise import compute_pairwise_metrics
    
    results = []
    
    # ICE-ID with temporal split + hard negatives
    print("\n--- ICE-ID (temporal split + hard negatives) ---")
    dataset = IceIdDataset(data_dir=str(RAW_DATA.parent))
    split = dataset.load()
    positive_pairs, cluster_labels = dataset.get_ground_truth()
    
    train_ids = split.metadata["train_record_ids"]
    test_ids = split.metadata["test_record_ids"]
    
    # Block within test
    blocker = TokenBlocker(fields=["nafn_norm", "parish"], min_token_length=2, max_block_size=200)
    test_df = split.records[split.records["id"].isin(test_ids)].copy()
    test_split = type(split)(name="test", records=test_df, cluster_labels={k: v for k, v in cluster_labels.items() if k in test_ids})
    candidates = blocker.block(test_split, record_ids=test_ids)
    
    positive_set = set(positive_pairs) | set((b, a) for a, b in positive_pairs)
    
    # Sample balanced pairs for training
    np.random.seed(42)
    train_df_records = split.records[split.records["id"].isin(train_ids)]
    train_blocker = TokenBlocker(fields=["nafn_norm"], min_token_length=2, max_block_size=100)
    train_split = type(split)(name="train", records=train_df_records, cluster_labels={k: v for k, v in cluster_labels.items() if k in train_ids})
    train_cands = train_blocker.block(train_split, record_ids=train_ids)
    
    train_data = []
    for p1, p2 in train_cands.pairs[:max_train_pairs * 10]:
        label = 1 if (p1, p2) in positive_set or (p2, p1) in positive_set else 0
        train_data.append({"id1": p1, "id2": p2, "label": label})
        if len(train_data) >= max_train_pairs:
            break
    train_df = pd.DataFrame(train_data)
    if not train_df.empty:
        pos = train_df[train_df["label"] == 1]
        neg_pool = train_df[train_df["label"] == 0]
        if len(pos) > 0 and len(neg_pool) > 0:
            neg = neg_pool.sample(n=min(len(pos) * 2, len(neg_pool)), random_state=42)
            train_df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    
    # Test pairs
    test_data = []
    for p1, p2 in candidates.pairs[:max_test_pairs * 10]:
        label = 1 if (p1, p2) in positive_set or (p2, p1) in positive_set else 0
        test_data.append({"id1": p1, "id2": p2, "label": label})
        if len(test_data) >= max_test_pairs:
            break
    
    pos_test = [d for d in test_data if d["label"] == 1]
    neg_test = [d for d in test_data if d["label"] == 0]
    neg_test = neg_test[:len(pos_test) * 2]
    test_data = pos_test + neg_test
    if len(test_data) > max_test_pairs:
        test_data = test_data[:max_test_pairs]
    np.random.shuffle(test_data)
    
    test_pairs = [(d["id1"], d["id2"]) for d in test_data]
    true_labels = np.array([d["label"] for d in test_data])
    
    # Train + score
    model = OpenNARSModel(preprocess="iceid")
    model.fit(split, train_df)
    scores = model.score(split, test_pairs)
    preds = (scores >= model.threshold).astype(int)
    
    metrics = compute_pairwise_metrics(true_labels, scores, preds)
    random_auc = compute_pairwise_metrics(true_labels, np.random.rand(len(true_labels)), (np.random.rand(len(true_labels)) >= 0.5).astype(int))["auc"]
    
    results.append({
        "dataset": "iceid",
        "protocol": "temporal_split_hard_neg",
        "auc": metrics["auc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "threshold": model.threshold,
        "n": len(test_pairs),
        "pos_rate": true_labels.mean() if len(true_labels) else 0.0,
        "random_auc": random_auc,
    })
    print(f"  ICE-ID: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
    
    # DeepMatcher datasets
    dm_datasets = ["abt_buy", "amazon_google", "dblp_acm", "dblp_scholar", "itunes_amazon", "walmart_amazon", "beer", "fodors_zagats"]
    
    for ds_name in dm_datasets:
        print(f"\n--- {ds_name} ---")
        try:
            dm = DeepMatcherDataset(ds_name, data_dir=str(DEEPMATCHER_DATA))
            dm_split = dm.load()
            
            # Use candset if available
            candset_path = DEEPMATCHER_DATA / ds_name / "candset_features_df.csv"
            if candset_path.exists():
                candset = pd.read_csv(candset_path)
                label_col = "gold" if "gold" in candset.columns else "label"
                
                # Stratified shuffle
                candset = candset.sample(frac=1, random_state=42)
                n = len(candset)
                train_n = int(n * 0.7)
                
                train_pairs_df = candset.iloc[:train_n][["ltable_id", "rtable_id", label_col]].copy()
                train_pairs_df.columns = ["id1", "id2", "label"]
                test_pairs_df = candset.iloc[train_n:][["ltable_id", "rtable_id", label_col]].copy()
                test_pairs_df.columns = ["id1", "id2", "label"]
                
                # Make IDs globally unique
                left_offset = 0
                right_offset = 1000000
                train_pairs_df["id1"] = train_pairs_df["id1"] + left_offset
                train_pairs_df["id2"] = train_pairs_df["id2"] + right_offset
                test_pairs_df["id1"] = test_pairs_df["id1"] + left_offset
                test_pairs_df["id2"] = test_pairs_df["id2"] + right_offset
                
                dm_split.left_table["id"] = dm_split.left_table["id"] + left_offset
                dm_split.right_table["id"] = dm_split.right_table["id"] + right_offset
                
                if len(train_pairs_df) > max_train_pairs:
                    train_pairs_df = train_pairs_df.iloc[:max_train_pairs].copy()
                if len(test_pairs_df) > max_test_pairs:
                    test_pairs_df = test_pairs_df.iloc[:max_test_pairs].copy()

                model = OpenNARSModel(preprocess="generic")
                model.fit(dm_split, train_pairs_df)
                
                test_pairs = list(zip(test_pairs_df["id1"], test_pairs_df["id2"]))
                true_labels = test_pairs_df["label"].values
                
                scores = model.score(dm_split, test_pairs)
                preds = (scores >= model.threshold).astype(int)
                
                metrics = compute_pairwise_metrics(true_labels, scores, preds)
                random_auc = compute_pairwise_metrics(true_labels, np.random.rand(len(true_labels)), (np.random.rand(len(true_labels)) >= 0.5).astype(int))["auc"]
                
                results.append({
                    "dataset": ds_name,
                    "protocol": "deepmatcher_split",
                    "auc": metrics["auc"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "threshold": model.threshold,
                    "n": len(test_pairs),
                    "pos_rate": true_labels.mean() if len(true_labels) else 0.0,
                    "random_auc": random_auc,
                })
                print(f"  {ds_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
            else:
                print(f"  Skipping {ds_name}: no candset")
        except Exception as e:
            print(f"  Error on {ds_name}: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "nars_rerun_all.csv", index=False)
    with open(ARTIFACTS_DIR / "nars_rerun_all.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nWrote {RESULTS_DIR / 'nars_rerun_all.csv'}")
    print(f"Wrote {ARTIFACTS_DIR / 'nars_rerun_all.json'}")


# =============================================================================
# NARS GRAPH EVALUATION (RANKING + CLUSTERING)
# =============================================================================

def run_nars_graph_eval():
    """Run NARS on ICE-ID candidate graph for ranking and clustering metrics."""
    from bench.data.iceid import IceIdDataset
    from bench.blocking.token_blocking import TokenBlocker
    from bench.models.opennars_adapter import OpenNARSModel
    from bench.metrics.pairwise import compute_pairwise_metrics
    from bench.clustering.connected_components import ConnectedComponentsClusterer
    
    print("\n--- NARS Graph Evaluation ---")
    
    dataset = IceIdDataset(data_dir=str(RAW_DATA.parent))
    split = dataset.load()
    positive_pairs, cluster_labels = dataset.get_ground_truth()
    
    # Sample subset for tractability
    np.random.seed(42)
    labeled_ids = [rid for rid, cid in cluster_labels.items() if cid != -1]
    sample_persons = list(set(cluster_labels[rid] for rid in labeled_ids))[:3000]
    sample_ids = [rid for rid, cid in cluster_labels.items() if cid in sample_persons]
    
    subset_df = split.records[split.records["id"].isin(sample_ids)].copy()
    subset_split = type(split)(
        name="iceid_subset",
        records=subset_df,
        cluster_labels={k: v for k, v in cluster_labels.items() if k in sample_ids}
    )
    
    # Block
    t0 = time.time()
    blocker = TokenBlocker(fields=["nafn_norm", "parish"], min_token_length=2, max_block_size=500)
    candidates = blocker.block(subset_split, record_ids=sample_ids)
    time_block = time.time() - t0
    rss_after_block = get_rss_mb()
    
    # Ground truth
    positive_set = set(positive_pairs) | set((b, a) for a, b in positive_pairs)
    subset_positive = set()
    person_to_ids = defaultdict(list)
    for rid in sample_ids:
        person_to_ids[cluster_labels[rid]].append(rid)
    for person, ids in person_to_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                subset_positive.add(tuple(sorted((ids[i], ids[j]))))
    
    # Labels for candidate pairs
    pair_labels = []
    for p1, p2 in candidates.pairs:
        label = 1 if tuple(sorted((p1, p2))) in subset_positive else 0
        pair_labels.append(label)
    pair_labels = np.array(pair_labels)
    
    # Train NARS
    train_data = []
    for i, (p1, p2) in enumerate(candidates.pairs[:10000]):
        train_data.append({"id1": p1, "id2": p2, "label": pair_labels[i]})
    train_df = pd.DataFrame(train_data)
    
    # Balance training
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0].sample(n=min(len(pos) * 2, len(train_df[train_df["label"] == 0])), random_state=42)
    train_df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    
    t0 = time.time()
    model = OpenNARSModel(preprocess="iceid")
    model.fit(subset_split, train_df)
    time_fit = time.time() - t0
    rss_after_fit = get_rss_mb()
    
    # Score all candidates
    t0 = time.time()
    max_pairs_scored = min(2000, len(candidates.pairs))
    pairs_scored = candidates.pairs[:max_pairs_scored]
    scores = model.score(subset_split, pairs_scored)
    time_score = time.time() - t0
    rss_after_score = get_rss_mb()
    
    pair_labels = pair_labels[:len(scores)]
    preds = (scores >= model.threshold).astype(int)
    
    # Pairwise metrics
    metrics = compute_pairwise_metrics(pair_labels, scores, preds)
    
    # Ranking: P@k and R@k
    k = int(pair_labels.sum())
    sorted_indices = np.argsort(scores)[::-1][:k]
    top_k_labels = pair_labels[sorted_indices]
    p_at_k = top_k_labels.mean()
    r_at_k = top_k_labels.sum() / pair_labels.sum() if pair_labels.sum() > 0 else 0
    
    # Clustering: connected components
    predicted_positive_edges = [(pairs_scored[i][0], pairs_scored[i][1], float(scores[i])) for i in range(len(preds)) if preds[i] == 1]
    clusterer = ConnectedComponentsClusterer()
    cluster_result = clusterer.cluster(predicted_positive_edges, sample_ids)
    
    pred_labels = [cluster_result.labels.get(rid, -1) for rid in sample_ids]
    true_labels_clustering = [cluster_labels.get(rid, -1) for rid in sample_ids]
    ari_cc = _adjusted_rand_index(true_labels_clustering, pred_labels)
    
    # B3 metrics
    def b3_metrics(true_labels, pred_labels, record_ids):
        prec_sum, rec_sum = 0.0, 0.0
        n = len(record_ids)
        
        true_clusters = defaultdict(set)
        pred_clusters = defaultdict(set)
        for i, rid in enumerate(record_ids):
            true_clusters[true_labels[i]].add(rid)
            pred_clusters[pred_labels[i]].add(rid)
        
        for i, rid in enumerate(record_ids):
            true_cluster = true_clusters[true_labels[i]]
            pred_cluster = pred_clusters[pred_labels[i]]
            intersection = len(true_cluster & pred_cluster)
            prec_sum += intersection / len(pred_cluster) if len(pred_cluster) > 0 else 0
            rec_sum += intersection / len(true_cluster) if len(true_cluster) > 0 else 0
        
        b3_p = prec_sum / n
        b3_r = rec_sum / n
        b3_f1 = 2 * b3_p * b3_r / (b3_p + b3_r) if (b3_p + b3_r) > 0 else 0
        return b3_p, b3_r, b3_f1
    
    b3_p, b3_r, b3_f1 = b3_metrics(true_labels_clustering, pred_labels, sample_ids)
    
    result = {
        "dataset": "iceid",
        "protocol": "candidate_graph_token_blocking",
        "subset_records": len(sample_ids),
        "subset_persons": len(sample_persons),
        "pairs_scored": len(pairs_scored),
        "pos_rate": float(pair_labels.mean()),
        "threshold": model.threshold,
        "auc": metrics["auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "p_at_k": p_at_k,
        "r_at_k": r_at_k,
        "ari_cc": ari_cc,
        "b3_precision": b3_p,
        "b3_recall": b3_r,
        "b3_f1": b3_f1,
        "time_block_s": time_block,
        "time_fit_s": time_fit,
        "time_score_s": time_score,
        "rss_mb_after_block": rss_after_block,
        "rss_mb_after_fit": rss_after_fit,
        "rss_mb_after_score": rss_after_score,
    }
    
    print(f"  Pairs scored: {len(candidates.pairs):,}")
    print(f"  AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"  P@k: {p_at_k:.4f}, R@k: {r_at_k:.4f}")
    print(f"  ARI-CC: {ari_cc:.6f}, B3-F1: {b3_f1:.4f}")
    
    pd.DataFrame([result]).to_csv(RESULTS_DIR / "nars_graph_eval.csv", index=False)
    with open(ARTIFACTS_DIR / "nars_graph_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nWrote {RESULTS_DIR / 'nars_graph_eval.csv'}")
    print(f"Wrote {ARTIFACTS_DIR / 'nars_graph_eval.json'}")


# =============================================================================
# NARS CALIBRATION SENSITIVITY
# =============================================================================

def run_nars_calibration():
    """Run NARS calibration sensitivity study."""
    from bench.data.iceid import IceIdDataset
    from bench.blocking.token_blocking import TokenBlocker
    from bench.models.opennars_adapter import OpenNARSModel
    from bench.metrics.pairwise import compute_pairwise_metrics
    
    print("\n--- NARS Calibration Sensitivity ---")
    
    dataset = IceIdDataset(data_dir=str(RAW_DATA.parent))
    split = dataset.load()
    positive_pairs, cluster_labels = dataset.get_ground_truth()
    
    # Sample subset
    np.random.seed(42)
    labeled_ids = [rid for rid, cid in cluster_labels.items() if cid != -1]
    sample_persons = list(set(cluster_labels[rid] for rid in labeled_ids))[:1000]
    sample_ids = [rid for rid, cid in cluster_labels.items() if cid in sample_persons]
    
    subset_df = split.records[split.records["id"].isin(sample_ids)].copy()
    subset_split = type(split)(
        name="iceid_subset",
        records=subset_df,
        cluster_labels={k: v for k, v in cluster_labels.items() if k in sample_ids}
    )
    
    # Block and get pairs
    blocker = TokenBlocker(fields=["nafn_norm"], min_token_length=2, max_block_size=200)
    candidates = blocker.block(subset_split, record_ids=sample_ids)
    
    positive_set = set(positive_pairs) | set((b, a) for a, b in positive_pairs)
    pair_labels = np.array([1 if (p1, p2) in positive_set or (p2, p1) in positive_set else 0 for p1, p2 in candidates.pairs])
    
    # Split into train/test
    n = len(candidates.pairs)
    train_n = int(n * 0.5)
    train_pairs = candidates.pairs[:train_n]
    train_labels = pair_labels[:train_n]
    test_pairs = candidates.pairs[train_n:]
    test_labels = pair_labels[train_n:]
    
    train_df = pd.DataFrame({"id1": [p[0] for p in train_pairs], "id2": [p[1] for p in train_pairs], "label": train_labels})
    
    # Train NARS
    model = OpenNARSModel(preprocess="iceid")
    model.fit(subset_split, train_df)
    scores = model.score(subset_split, test_pairs)
    
    # Test different thresholds
    results = []
    for strategy, threshold in [
        ("fixed_0.5", 0.5),
        ("fixed_0.3", 0.3),
        ("median_midpoint", 0.5),
        ("optimal_f1", None),
    ]:
        if threshold is None:
            # Find optimal
            best_f1, best_thresh = 0, 0.5
            for t in np.linspace(0.1, 0.9, 17):
                preds = (scores >= t).astype(int)
                m = compute_pairwise_metrics(test_labels, scores, preds)
                if m["f1"] > best_f1:
                    best_f1, best_thresh = m["f1"], t
            threshold = best_thresh
        
        preds = (scores >= threshold).astype(int)
        metrics = compute_pairwise_metrics(test_labels, scores, preds)
        
        results.append({
            "strategy": strategy,
            "threshold": threshold,
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
        print(f"  {strategy}: threshold={threshold:.2f}, F1={metrics['f1']:.4f}")
    
    pd.DataFrame(results).to_csv(RESULTS_DIR / "nars_calibration_sensitivity.csv", index=False)
    with open(ARTIFACTS_DIR / "nars_calibration_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nWrote {RESULTS_DIR / 'nars_calibration_sensitivity.csv'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run ICE-ID benchmark experiments")
    parser.add_argument("experiment", nargs="?", default="all",
                        choices=["all", "nars-rerun", "nars-graph", "nars-calibration"],
                        help="Which experiment to run")
    parser.add_argument("--max-train-pairs", type=int, default=2000, help="Max training pairs per dataset")
    parser.add_argument("--max-test-pairs", type=int, default=2000, help="Max test pairs per dataset")
    args = parser.parse_args()
    
    print("=" * 70)
    print("ICE-ID BENCHMARK EXPERIMENTS")
    print("=" * 70)
    
    if args.experiment in ["all", "nars-rerun"]:
        run_nars_rerun(max_train_pairs=int(args.max_train_pairs), max_test_pairs=int(args.max_test_pairs))
    
    if args.experiment in ["all", "nars-graph"]:
        run_nars_graph_eval()
    
    if args.experiment in ["all", "nars-calibration"]:
        run_nars_calibration()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

