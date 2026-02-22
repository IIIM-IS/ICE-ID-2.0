#!/usr/bin/env python3
"""
Comprehensive NARS evaluation script.

Computes all metrics for Table 6 of the NARS paper:
- Pairwise: P, R, F1, Acc, AUC
- Threshold
- Ranking: P@k, R@k
- Clustering: ARI-CC, ARI-AG

Usage:
    python scripts/run_nars_full_eval.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.data.iceid import IceIdDataset
from bench.data.deepmatcher import DeepMatcherDataset
from bench.models.nars import NarsModel
from bench.blocking.token_blocking import TokenBlocker
from bench.metrics.pairwise import compute_pairwise_metrics
from bench.metrics.ranking import compute_ranking_metrics
from bench.metrics.clustering import compute_clustering_metrics, _adjusted_rand_index
from bench.clustering.connected_components import ConnectedComponentsClusterer

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "benchmark_results"
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts"


def build_clusters_from_pairs(
    pairs: List[Tuple[str, str]],
    labels: np.ndarray
) -> Dict[str, int]:
    """
    Build cluster labels from pairwise ground truth for two-table datasets.
    
    Uses connected components on positive pairs to define clusters.
    """
    from collections import deque
    
    positive_pairs = [(pairs[i][0], pairs[i][1]) for i in range(len(pairs)) if labels[i] == 1]
    
    all_nodes = set()
    for p1, p2 in pairs:
        all_nodes.add(p1)
        all_nodes.add(p2)
    
    if not positive_pairs:
        return {n: i for i, n in enumerate(all_nodes)}
    
    adj = defaultdict(set)
    for p1, p2 in positive_pairs:
        adj[p1].add(p2)
        adj[p2].add(p1)
    
    visited = set()
    cluster_labels = {}
    cluster_id = 0
    
    for node in all_nodes:
        if node in visited:
            continue
        queue = deque([node])
        while queue:
            n = queue.popleft()
            if n in visited:
                continue
            visited.add(n)
            cluster_labels[n] = cluster_id
            for neighbor in adj.get(n, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        cluster_id += 1
    
    return cluster_labels


def compute_ari_from_pairs(
    pairs: List[Tuple[str, str]],
    scores: np.ndarray,
    true_labels: Dict[str, int],
    threshold: float,
    method: str = "cc"
) -> float:
    """
    Compute ARI from scored pairs using thresholding + clustering.
    
    Args:
        pairs: List of (id1, id2) pairs.
        scores: Scores for each pair.
        true_labels: Ground truth cluster labels.
        threshold: Score threshold for edge inclusion.
        method: "cc" for connected components, "ag" for agglomerative.
    
    Returns:
        ARI score.
    """
    edges_above_threshold = [
        (pairs[i][0], pairs[i][1])
        for i in range(len(pairs))
        if scores[i] >= threshold
    ]
    
    if not edges_above_threshold:
        return 0.0
    
    all_nodes = set()
    for p1, p2 in edges_above_threshold:
        all_nodes.add(p1)
        all_nodes.add(p2)
    
    from collections import deque
    
    if method == "cc":
        adj = defaultdict(set)
        for p1, p2 in edges_above_threshold:
            adj[p1].add(p2)
            adj[p2].add(p1)
        
        visited = set()
        pred_clusters = []
        
        for node in all_nodes:
            if node in visited:
                continue
            cluster = []
            queue = deque([node])
            while queue:
                n = queue.popleft()
                if n in visited:
                    continue
                visited.add(n)
                cluster.append(n)
                for neighbor in adj[n]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            pred_clusters.append(cluster)
    elif method == "ag":
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            node_list = sorted(all_nodes)
            node_idx = {n: i for i, n in enumerate(node_list)}
            n = len(node_list)
            
            if n < 2:
                return 0.0
            
            dist_matrix = np.ones((n, n))
            np.fill_diagonal(dist_matrix, 0)
            
            for i, (p1, p2) in enumerate(pairs):
                if p1 in node_idx and p2 in node_idx:
                    similarity = scores[i]
                    distance = 1.0 - similarity
                    i1, i2 = node_idx[p1], node_idx[p2]
                    dist_matrix[i1, i2] = distance
                    dist_matrix[i2, i1] = distance
            
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method='average')
            
            cluster_assignments = fcluster(Z, t=1 - threshold, criterion='distance')
            
            pred_clusters = defaultdict(list)
            for i, c in enumerate(cluster_assignments):
                pred_clusters[c].append(node_list[i])
            pred_clusters = list(pred_clusters.values())
        except Exception:
            pred_clusters = [[n] for n in all_nodes]
    else:
        pred_clusters = [[n] for n in all_nodes]
    
    common_nodes = [n for n in all_nodes if n in true_labels]
    if len(common_nodes) < 2:
        return 0.0
    
    true_arr = [true_labels[n] for n in common_nodes]
    
    pred_label_map = {}
    for cluster_id, members in enumerate(pred_clusters):
        for member in members:
            pred_label_map[member] = cluster_id
    
    pred_arr = [pred_label_map.get(n, -1) for n in common_nodes]
    
    return _adjusted_rand_index(true_arr, pred_arr)


def select_threshold_by_f1(
    scores: np.ndarray,
    labels: np.ndarray,
    default: float = 0.5
) -> float:
    """Select a threshold that maximizes F1 on validation data."""
    if scores.size == 0 or labels.size == 0:
        return default
    if np.sum(labels == 1) == 0 or np.sum(labels == 0) == 0:
        return default
    best_f1 = -1.0
    best_threshold = default
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (scores >= threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        if f1 > best_f1 or (abs(f1 - best_f1) < 1e-10 and threshold > best_threshold):
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def evaluate_nars_on_dataset(
    dataset_name: str,
    dataset_provider: Any,
    max_pairs: int = 50000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run full NARS evaluation on a dataset.
    
    Args:
        dataset_name: Name for reporting.
        dataset_provider: Dataset object with load() method.
        max_pairs: Maximum pairs to evaluate.
        seed: Random seed.
    
    Returns:
        Dict with all metrics.
    """
    np.random.seed(seed)
    print(f"\n  Evaluating NARS on {dataset_name}...")
    
    split = dataset_provider.load()
    
    is_iceid = hasattr(split, 'cluster_labels') and split.cluster_labels
    
    if is_iceid:
        records_df = split.records
        cluster_labels = split.cluster_labels
        
        person_to_ids = defaultdict(list)
        for _, row in records_df.iterrows():
            rid = row["id"]
            if rid in cluster_labels:
                person_to_ids[cluster_labels[rid]].append(rid)
        
        multi_record_persons = [p for p, ids in person_to_ids.items() if len(ids) >= 2]
        np.random.shuffle(multi_record_persons)
        selected_persons = multi_record_persons[:min(2000, len(multi_record_persons))]
        
        sample_ids = []
        for person in selected_persons:
            sample_ids.extend(person_to_ids[person])
        
        sample_df = records_df[records_df["id"].isin(sample_ids)].copy()
        
        positive_pairs = []
        for person in selected_persons:
            ids = person_to_ids[person]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    positive_pairs.append((ids[i], ids[j], 1))
        
        n_pos = len(positive_pairs)
        n_neg = min(n_pos * 2, max_pairs - n_pos)
        
        negative_pairs = []
        id_list = sample_ids
        attempts = 0
        while len(negative_pairs) < n_neg and attempts < n_neg * 10:
            i, j = np.random.choice(len(id_list), 2, replace=False)
            id1, id2 = id_list[i], id_list[j]
            if cluster_labels.get(id1) != cluster_labels.get(id2):
                negative_pairs.append((id1, id2, 0))
            attempts += 1
        
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        train_split = int(len(all_pairs) * 0.7)
        val_split = int(len(all_pairs) * 0.85)
        
        train_pairs = all_pairs[:train_split]
        val_pairs = all_pairs[train_split:val_split]
        test_pairs = all_pairs[val_split:]
        
        records_dict = {r["id"]: dict(r) for _, r in sample_df.iterrows()}
        
    else:
        left_df = split.left_table
        right_df = split.right_table
        
        left_records = {f"L_{r['id']}": dict(r) for _, r in left_df.iterrows()}
        right_records = {f"R_{r['id']}": dict(r) for _, r in right_df.iterrows()}
        records_dict = {**left_records, **right_records}
        
        cluster_labels = {}
        
        train_df = split.train_pairs
        val_df = split.val_pairs
        test_df = split.test_pairs
        
        def df_to_pairs(df):
            if df is None:
                return []
            pairs = []
            for _, row in df.iterrows():
                lid = f"L_{row['ltable_id']}" if not str(row['ltable_id']).startswith('L_') else row['ltable_id']
                rid = f"R_{row['rtable_id']}" if not str(row['rtable_id']).startswith('R_') else row['rtable_id']
                label = int(row.get('label', 0))
                pairs.append((lid, rid, label))
            return pairs
        
        train_pairs = df_to_pairs(train_df)
        val_pairs = df_to_pairs(val_df)
        test_pairs = df_to_pairs(test_df)
    
    print(f"    Train: {len(train_pairs)} pairs ({sum(p[2] for p in train_pairs)} pos)")
    print(f"    Val: {len(val_pairs)} pairs ({sum(p[2] for p in val_pairs)} pos)")
    print(f"    Test: {len(test_pairs)} pairs ({sum(p[2] for p in test_pairs)} pos)")
    
    if is_iceid:
        preprocess = "iceid"
    else:
        dataset_key = getattr(dataset_provider, "name", "")
        preprocess = f"deepmatcher:{dataset_key}" if dataset_key else "deepmatcher"
    model = NarsModel(preprocess=preprocess)
    
    start_time = time.time()
    for id1, id2, label in train_pairs:
        rec1 = records_dict.get(id1)
        rec2 = records_dict.get(id2)
        if rec1 is not None and rec2 is not None:
            model.fit_pair(rec1, rec2, label)
    
    val_scores = []
    val_labels = []
    for id1, id2, label in val_pairs:
        rec1 = records_dict.get(id1)
        rec2 = records_dict.get(id2)
        if rec1 is not None and rec2 is not None:
            score = model.score_pair(rec1, rec2)
            val_scores.append(score)
            val_labels.append(label)
    
    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    
    pos_scores = val_scores[val_labels == 1]
    neg_scores = val_scores[val_labels == 0]
    
    if is_iceid:
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
        else:
            threshold = 0.5
    else:
        threshold = select_threshold_by_f1(val_scores, val_labels, default=0.5)
    
    model.threshold = threshold
    fit_time = time.time() - start_time
    
    start_time = time.time()
    test_scores = []
    test_labels = []
    test_pair_ids = []
    
    for id1, id2, label in test_pairs:
        rec1 = records_dict.get(id1)
        rec2 = records_dict.get(id2)
        if rec1 is not None and rec2 is not None:
            score = model.score_pair(rec1, rec2)
            test_scores.append(score)
            test_labels.append(label)
            test_pair_ids.append((id1, id2))
    
    score_time = time.time() - start_time
    
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)
    
    pairwise = compute_pairwise_metrics(test_labels, y_scores=test_scores, threshold=threshold)
    
    n_positives = int((test_labels == 1).sum())
    if n_positives > 0:
        ranking = compute_ranking_metrics(test_labels, test_scores, k_values=[n_positives])
        p_at_k = ranking.get(f"p_at_{n_positives}", 0.0)
        r_at_k = ranking.get(f"r_at_{n_positives}", 0.0)
    else:
        p_at_k = 0.0
        r_at_k = 0.0
    
    if is_iceid and cluster_labels:
        ari_cc = compute_ari_from_pairs(test_pair_ids, test_scores, cluster_labels, threshold, "cc")
        ari_ag = compute_ari_from_pairs(test_pair_ids, test_scores, cluster_labels, threshold, "ag")
    else:
        true_cluster_labels = build_clusters_from_pairs(test_pair_ids, test_labels)
        ari_cc = compute_ari_from_pairs(test_pair_ids, test_scores, true_cluster_labels, threshold, "cc")
        ari_ag = compute_ari_from_pairs(test_pair_ids, test_scores, true_cluster_labels, threshold, "ag")
    
    results = {
        "dataset": dataset_name,
        "precision": round(pairwise["precision"], 4),
        "recall": round(pairwise["recall"], 4),
        "f1": round(pairwise["f1"], 4),
        "accuracy": round(pairwise["accuracy"], 4),
        "threshold": round(threshold, 4),
        "auc": round(pairwise.get("auc", 0.0), 4),
        "ari_cc": round(ari_cc, 4),
        "ari_ag": round(ari_ag, 4),
        "p_at_k": round(p_at_k, 4),
        "r_at_k": round(r_at_k, 4),
        "n_test_pairs": len(test_pairs),
        "n_positives": n_positives,
        "pos_rate": round(n_positives / len(test_pairs), 4) if test_pairs else 0,
        "fit_time_s": round(fit_time, 2),
        "score_time_s": round(score_time, 2),
    }
    
    print(f"    P={results['precision']:.3f} R={results['recall']:.3f} F1={results['f1']:.3f} "
          f"Acc={results['accuracy']:.3f} AUC={results['auc']:.3f}")
    print(f"    P@k={results['p_at_k']:.3f} R@k={results['r_at_k']:.3f} "
          f"ARI-CC={results['ari_cc']:.4f}")
    
    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("NARS FULL EVALUATION")
    print("=" * 70)
    
    results = []
    
    print("\n1. ICE-ID Dataset")
    try:
        raw_data_path = BASE_DIR.parent / "raw_data"
        if not raw_data_path.exists():
            raw_data_path = BASE_DIR.parent / "data" / "raw_data"
        iceid = IceIdDataset(data_dir=str(raw_data_path))
        iceid_results = evaluate_nars_on_dataset("ICE-ID", iceid)
        results.append(iceid_results)
    except Exception as e:
        import traceback
        print(f"  ICE-ID failed: {e}")
        traceback.print_exc()
    
    dm_datasets = [
        "abt_buy", "amazon_google", "dblp_acm", "dblp_scholar",
        "itunes_amazon", "walmart_amazon", "beer", "fodors_zagats"
    ]
    
    print("\n2. Classic ER Datasets")
    for ds_name in dm_datasets:
        try:
            dm = DeepMatcherDataset(ds_name, data_dir=str(BASE_DIR / "deepmatcher_data"))
            ds_results = evaluate_nars_on_dataset(ds_name.upper().replace("_", "-"), dm)
            results.append(ds_results)
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
    
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "nars_full_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    json_path = ARTIFACTS_DIR / "nars_full_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Artifacts saved to {json_path}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (for Table 6)")
    print("=" * 70)
    print(f"{'Dataset':<15} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6} {'Thr':>6} {'AUC':>6} {'ARI-CC':>8} {'P@k':>6} {'R@k':>6}")
    print("-" * 85)
    for r in results:
        print(f"{r['dataset']:<15} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
              f"{r['accuracy']:>6.3f} {r['threshold']:>6.2f} {r['auc']:>6.3f} "
              f"{r['ari_cc']:>8.4f} {r['p_at_k']:>6.3f} {r['r_at_k']:>6.3f}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
