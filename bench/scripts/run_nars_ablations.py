#!/usr/bin/env python3
"""
NARS ablation study using the same protocol as run_nars_full_eval.py.

Reruns the ICE-ID evaluation with each judgment category excluded,
so absolute F1 values are directly comparable to the main result.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel
from bench.metrics.pairwise import compute_pairwise_metrics

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "benchmark_results"


def select_threshold_by_f1(scores, labels, default=0.5):
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


def run_ablation(
    exclude_name: str,
    exclude_judgments: List[str],
    records_dict: Dict,
    cluster_labels: Dict,
    train_pairs: List,
    val_pairs: List,
    test_pairs: List,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a single ablation experiment."""
    np.random.seed(seed)
    print(f"\n  Ablation: {exclude_name} (exclude={exclude_judgments})")

    model = NarsModel(preprocess="iceid", exclude_judgments=exclude_judgments)

    start = time.time()
    for id1, id2, label in train_pairs:
        rec1 = records_dict.get(id1)
        rec2 = records_dict.get(id2)
        if rec1 is not None and rec2 is not None:
            model.fit_pair(rec1, rec2, label)

    # Calibrate threshold on validation pairs (same as run_nars_full_eval)
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
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
    else:
        threshold = 0.5

    # Score test pairs
    test_scores = []
    test_labels = []
    for id1, id2, label in test_pairs:
        rec1 = records_dict.get(id1)
        rec2 = records_dict.get(id2)
        if rec1 is not None and rec2 is not None:
            score = model.score_pair(rec1, rec2)
            test_scores.append(score)
            test_labels.append(label)

    elapsed = time.time() - start
    test_scores = np.array(test_scores)
    test_labels = np.array(test_labels)

    pairwise = compute_pairwise_metrics(test_labels, y_scores=test_scores, threshold=threshold)

    result = {
        "ablation": exclude_name,
        "f1": pairwise["f1"],
        "precision": pairwise["precision"],
        "recall": pairwise["recall"],
        "auc": pairwise.get("auc", 0.0),
        "accuracy": pairwise["accuracy"],
        "threshold": threshold,
        "time_s": round(elapsed, 2),
        "n_test": len(test_scores),
        "n_pos": int((test_labels == 1).sum()),
    }
    print(f"    F1={result['f1']:.4f}  P={result['precision']:.4f}  "
          f"R={result['recall']:.4f}  AUC={result['auc']:.4f}  thr={threshold:.3f}")
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NARS ABLATION STUDY (same protocol as run_nars_full_eval.py)")
    print("=" * 70)

    # === Load dataset (same as run_nars_full_eval.py) ===
    raw_data_path = BASE_DIR.parent / "raw_data"
    if not raw_data_path.exists():
        raw_data_path = BASE_DIR.parent / "data" / "raw_data"
    iceid = IceIdDataset(data_dir=str(raw_data_path))
    split = iceid.load()

    records_df = split.records
    cluster_labels = split.cluster_labels

    # === Build pairs (same as run_nars_full_eval.py) ===
    np.random.seed(42)
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

    max_pairs = 50000
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

    print(f"\nPairs: {len(all_pairs)} total "
          f"(train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)})")
    print(f"Positives in test: {sum(1 for _, _, l in test_pairs if l == 1)}")

    # === Define ablations ===
    ablations = [
        ("Full model", []),
        ("- Name judgments", ["nafn_norm", "first_name", "patronym", "surname"]),
        ("- Birthyear judgments", ["birthyear"]),
        ("- Sex judgments", ["sex"]),
        ("- Geographic judgments", ["farm", "parish", "district", "county"]),
        ("- Census year (heimild)", ["heimild"]),
    ]

    # === Run ablations ===
    results = []
    for name, exclude in ablations:
        result = run_ablation(
            name, exclude, records_dict, cluster_labels,
            train_pairs, val_pairs, test_pairs,
        )
        results.append(result)

    # Compute delta_f1
    full_f1 = results[0]["f1"]
    for r in results:
        r["delta_f1"] = r["f1"] - full_f1

    # === Save results ===
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "nars_ablations_consistent.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Ablation':<30} {'F1':>8} {'dF1':>8} {'P':>8} {'R':>8} {'AUC':>8} {'Thr':>6}")
    print("-" * 80)
    for r in results:
        delta = f"{r['delta_f1']:+.4f}" if r['delta_f1'] != 0 else "---"
        print(f"{r['ablation']:<30} {r['f1']:>8.4f} {delta:>8} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['auc']:>8.4f} {r['threshold']:>6.3f}")


if __name__ == "__main__":
    main()
