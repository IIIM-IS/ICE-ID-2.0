#!/usr/bin/env python3
"""
Comprehensive evaluation of Fellegi-Sunter and Rules models.

Uses the same protocol as run_nars_full_eval.py for fair comparison:
- ICE-ID: 2:1 neg:pos, 50k cap, 70/15/15 split, median-midpoint threshold
- Classic ER: existing train/val/test splits, F1-optimal threshold

Also attempts Ditto on ICE-ID if PyTorch is available.

Usage:
    python scripts/run_baseline_eval.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.data.iceid import IceIdDataset
from bench.data.deepmatcher import DeepMatcherDataset
from bench.models.fellegi_sunter import FellegiSunterModel
from bench.models.rules import RulesModel
from bench.core.types import DatasetSplit
from bench.metrics.pairwise import compute_pairwise_metrics

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "benchmark_results"


def select_threshold_by_f1(scores, labels, default=0.5):
    """Select threshold that maximizes F1 on given data."""
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


def build_iceid_pairs(dataset_split, max_pairs=50000, seed=42):
    """
    Build ICE-ID pairs using same protocol as run_nars_full_eval.py.
    Returns (train_pairs, val_pairs, test_pairs) as lists of (id1, id2, label).
    """
    np.random.seed(seed)

    records_df = dataset_split.records
    cluster_labels = dataset_split.cluster_labels

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

    return train_pairs, val_pairs, test_pairs, sample_ids


def pairs_to_df(pairs):
    """Convert list of (id1, id2, label) to DataFrame."""
    return pd.DataFrame(pairs, columns=["id1", "id2", "label"])


def evaluate_fs_on_iceid(max_pairs=50000, seed=42):
    """Evaluate Fellegi-Sunter on ICE-ID with proper protocol."""
    print("\n  Evaluating Fellegi-Sunter on ICE-ID...")

    raw_data_path = BASE_DIR.parent / "raw_data"
    iceid = IceIdDataset(data_dir=str(raw_data_path))
    split = iceid.load()

    # Build pairs using same protocol as NARS eval
    train_pairs, val_pairs, test_pairs, sample_ids = build_iceid_pairs(
        split, max_pairs=max_pairs, seed=seed
    )
    print(f"    Train: {len(train_pairs)} pairs ({sum(p[2] for p in train_pairs)} pos)")
    print(f"    Val: {len(val_pairs)} pairs ({sum(p[2] for p in val_pairs)} pos)")
    print(f"    Test: {len(test_pairs)} pairs ({sum(p[2] for p in test_pairs)} pos)")

    # Create a clean DatasetSplit without 'person' column (prevents label leakage)
    clean_records = split.records[split.records["id"].isin(sample_ids)].copy()
    cols_to_drop = [c for c in ["person"] if c in clean_records.columns]
    clean_records = clean_records.drop(columns=cols_to_drop)

    clean_split = DatasetSplit(
        name="iceid",
        records=clean_records,
        cluster_labels=split.cluster_labels,
    )

    train_df = pairs_to_df(train_pairs)
    val_df = pairs_to_df(val_pairs)

    # Fit model
    model = FellegiSunterModel()
    start_time = time.time()
    model.fit(clean_split, train_df, val_df)
    fit_time = time.time() - start_time

    # Score validation pairs for threshold calibration
    val_pair_tuples = [(p[0], p[1]) for p in val_pairs]
    val_labels = np.array([p[2] for p in val_pairs])
    val_scores = model.score(clean_split, val_pair_tuples)

    # ICE-ID uses median midpoint threshold
    pos_scores = val_scores[val_labels == 1]
    neg_scores = val_scores[val_labels == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
    else:
        threshold = 0.5

    # Score test pairs
    test_pair_tuples = [(p[0], p[1]) for p in test_pairs]
    test_labels = np.array([p[2] for p in test_pairs])
    start_time = time.time()
    test_scores = model.score(clean_split, test_pair_tuples)
    score_time = time.time() - start_time

    # Compute metrics
    pairwise = compute_pairwise_metrics(test_labels, y_scores=test_scores, threshold=threshold)

    results = {
        "model": "fellegi_sunter",
        "dataset": "ICE-ID",
        "precision": round(pairwise["precision"], 4),
        "recall": round(pairwise["recall"], 4),
        "f1": round(pairwise["f1"], 4),
        "accuracy": round(pairwise["accuracy"], 4),
        "threshold": round(threshold, 4),
        "auc": round(pairwise.get("auc", 0.0), 4),
        "n_test_pairs": len(test_pairs),
        "n_positives": int((test_labels == 1).sum()),
        "fit_time_s": round(fit_time, 2),
        "score_time_s": round(score_time, 2),
        "status": "completed",
    }

    print(f"    P={results['precision']:.3f} R={results['recall']:.3f} F1={results['f1']:.3f} "
          f"Acc={results['accuracy']:.3f} AUC={results['auc']:.3f} Thr={results['threshold']:.3f}")

    return results


def evaluate_rules_on_iceid(max_pairs=50000, seed=42):
    """Evaluate Rules model on ICE-ID with proper protocol."""
    print("\n  Evaluating Rules on ICE-ID...")

    raw_data_path = BASE_DIR.parent / "raw_data"
    iceid = IceIdDataset(data_dir=str(raw_data_path))
    split = iceid.load()

    # Build pairs using same protocol as NARS eval
    train_pairs, val_pairs, test_pairs, sample_ids = build_iceid_pairs(
        split, max_pairs=max_pairs, seed=seed
    )
    print(f"    Train: {len(train_pairs)} pairs ({sum(p[2] for p in train_pairs)} pos)")
    print(f"    Val: {len(val_pairs)} pairs ({sum(p[2] for p in val_pairs)} pos)")
    print(f"    Test: {len(test_pairs)} pairs ({sum(p[2] for p in test_pairs)} pos)")

    # Create a clean DatasetSplit without 'person' column
    clean_records = split.records[split.records["id"].isin(sample_ids)].copy()
    cols_to_drop = [c for c in ["person"] if c in clean_records.columns]
    clean_records = clean_records.drop(columns=cols_to_drop)

    clean_split = DatasetSplit(
        name="iceid",
        records=clean_records,
        cluster_labels=split.cluster_labels,
    )

    # Rules doesn't need training but we fit() for interface consistency
    model = RulesModel()
    model.fit(clean_split, pairs_to_df(train_pairs))

    # Score validation pairs for threshold calibration
    val_pair_tuples = [(p[0], p[1]) for p in val_pairs]
    val_labels = np.array([p[2] for p in val_pairs])
    val_scores = model.score(clean_split, val_pair_tuples)

    # ICE-ID uses median midpoint threshold
    pos_scores = val_scores[val_labels == 1]
    neg_scores = val_scores[val_labels == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
    else:
        threshold = 0.5

    # Score test pairs
    test_pair_tuples = [(p[0], p[1]) for p in test_pairs]
    test_labels = np.array([p[2] for p in test_pairs])
    start_time = time.time()
    test_scores = model.score(clean_split, test_pair_tuples)
    score_time = time.time() - start_time

    pairwise = compute_pairwise_metrics(test_labels, y_scores=test_scores, threshold=threshold)

    results = {
        "model": "rules",
        "dataset": "ICE-ID",
        "precision": round(pairwise["precision"], 4),
        "recall": round(pairwise["recall"], 4),
        "f1": round(pairwise["f1"], 4),
        "accuracy": round(pairwise["accuracy"], 4),
        "threshold": round(threshold, 4),
        "auc": round(pairwise.get("auc", 0.0), 4),
        "n_test_pairs": len(test_pairs),
        "n_positives": int((test_labels == 1).sum()),
        "fit_time_s": 0.0,
        "score_time_s": round(score_time, 2),
        "status": "completed",
    }

    print(f"    P={results['precision']:.3f} R={results['recall']:.3f} F1={results['f1']:.3f} "
          f"Acc={results['accuracy']:.3f} AUC={results['auc']:.3f} Thr={results['threshold']:.3f}")

    return results


def evaluate_fs_on_deepmatcher(dataset_name):
    """Evaluate Fellegi-Sunter on a DeepMatcher dataset."""
    print(f"\n  Evaluating Fellegi-Sunter on {dataset_name}...")

    dm = DeepMatcherDataset(dataset_name, data_dir=str(BASE_DIR / "deepmatcher_data"))
    split = dm.load()

    if split.train_pairs is None or split.test_pairs is None:
        print(f"    No train/test pairs for {dataset_name}")
        return None

    train_pairs = split.train_pairs
    val_pairs = split.val_pairs
    test_pairs = split.test_pairs

    id1_col = "ltable_id" if "ltable_id" in train_pairs.columns else "id1"
    id2_col = "rtable_id" if "rtable_id" in train_pairs.columns else "id2"

    n_train = len(train_pairs)
    n_train_pos = int(train_pairs["label"].sum()) if "label" in train_pairs.columns else 0
    n_test = len(test_pairs)
    n_test_pos = int(test_pairs["label"].sum()) if "label" in test_pairs.columns else 0
    print(f"    Train: {n_train} pairs ({n_train_pos} pos)")
    if val_pairs is not None:
        print(f"    Val: {len(val_pairs)} pairs ({int(val_pairs['label'].sum())} pos)")
    print(f"    Test: {n_test} pairs ({n_test_pos} pos)")

    model = FellegiSunterModel()
    start_time = time.time()
    model.fit(split, train_pairs, val_pairs)
    fit_time = time.time() - start_time

    # Score validation pairs for F1-optimal threshold
    if val_pairs is not None and len(val_pairs) > 0:
        val_pair_tuples = list(zip(
            val_pairs[id1_col].astype(int).tolist(),
            val_pairs[id2_col].astype(int).tolist()
        ))
        val_labels = val_pairs["label"].astype(int).values
        val_scores = model.score(split, val_pair_tuples)
        threshold = select_threshold_by_f1(val_scores, val_labels)
    else:
        threshold = model.threshold

    # Score test pairs
    test_pair_tuples = list(zip(
        test_pairs[id1_col].astype(int).tolist(),
        test_pairs[id2_col].astype(int).tolist()
    ))
    test_labels = test_pairs["label"].astype(int).values

    start_time = time.time()
    test_scores = model.score(split, test_pair_tuples)
    score_time = time.time() - start_time

    pairwise = compute_pairwise_metrics(test_labels, y_scores=test_scores, threshold=threshold)

    ds_display = dataset_name.upper().replace("_", "-")
    results = {
        "model": "fellegi_sunter",
        "dataset": ds_display,
        "precision": round(pairwise["precision"], 4),
        "recall": round(pairwise["recall"], 4),
        "f1": round(pairwise["f1"], 4),
        "accuracy": round(pairwise["accuracy"], 4),
        "threshold": round(threshold, 4),
        "auc": round(pairwise.get("auc", 0.0), 4),
        "n_test_pairs": n_test,
        "n_positives": n_test_pos,
        "fit_time_s": round(fit_time, 2),
        "score_time_s": round(score_time, 2),
        "status": "completed",
    }

    print(f"    P={results['precision']:.3f} R={results['recall']:.3f} F1={results['f1']:.3f} "
          f"Acc={results['accuracy']:.3f} AUC={results['auc']:.3f} Thr={results['threshold']:.3f}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)

    all_results = []

    # 1. Fellegi-Sunter on ICE-ID
    print("\n1. Fellegi-Sunter on ICE-ID")
    try:
        fs_iceid = evaluate_fs_on_iceid()
        all_results.append(fs_iceid)
    except Exception as e:
        import traceback
        print(f"  FS ICE-ID failed: {e}")
        traceback.print_exc()

    # 2. Rules on ICE-ID
    print("\n2. Rules on ICE-ID")
    try:
        rules_iceid = evaluate_rules_on_iceid()
        all_results.append(rules_iceid)
    except Exception as e:
        import traceback
        print(f"  Rules ICE-ID failed: {e}")
        traceback.print_exc()

    # 3. Fellegi-Sunter on classic ER datasets
    dm_datasets = [
        "abt_buy", "amazon_google", "dblp_acm", "dblp_scholar",
        "itunes_amazon", "walmart_amazon", "beer", "fodors_zagats"
    ]

    print("\n3. Fellegi-Sunter on Classic ER Datasets")
    for ds_name in dm_datasets:
        try:
            fs_result = evaluate_fs_on_deepmatcher(ds_name)
            if fs_result:
                all_results.append(fs_result)
        except Exception as e:
            import traceback
            print(f"  FS {ds_name} failed: {e}")
            traceback.print_exc()

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "baseline_eval.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Dataset':<20} {'P':>6} {'R':>6} {'F1':>6} {'AUC':>6} {'Thr':>6}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<20} {r['dataset']:<20} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['auc']:>6.3f} {r['threshold']:>6.3f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
