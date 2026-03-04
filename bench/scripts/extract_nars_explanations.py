#!/usr/bin/env python3
"""
Extract NARS judgment weights and per-pair explanations for the paper.

Outputs:
  - paper_artifacts/nars_judgment_weights.json
  - paper_artifacts/nars_pair_explanations.json
  - papers/figures/nars_pair_explanation.pdf (waterfall chart)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel

BASE_DIR = Path(__file__).parent.parent
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts"
FIGURES_DIR = BASE_DIR.parent / "papers" / "figures"

ARTIFACTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    np.random.seed(42)

    # --- Load ICE-ID ---
    print("Loading ICE-ID dataset...")
    dataset = IceIdDataset(data_dir=str(BASE_DIR.parent / "raw_data"))
    split = dataset.load()

    records_df = split.records
    cluster_labels = split.cluster_labels

    # Build person → record-id mapping
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

    # Build pairs (same protocol as run_nars_full_eval.py)
    positive_pairs = []
    for person in selected_persons:
        ids = person_to_ids[person]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                positive_pairs.append((ids[i], ids[j], 1))

    n_pos = len(positive_pairs)
    n_neg = min(n_pos * 2, 50000 - n_pos)

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

    train_pairs_list = all_pairs[:train_split]
    test_pairs_list = all_pairs[val_split:]

    train_df = pd.DataFrame(train_pairs_list, columns=["id1", "id2", "label"])

    records_dict = {r["id"]: dict(r) for _, r in records_df.iterrows()}

    # --- Train NARS ---
    print(f"Training NARS on {len(train_df)} pairs...")
    model = NarsModel(preprocess="iceid")
    model.fit(split, train_df)

    # --- Extract judgment weights ---
    print("Extracting judgment weights...")
    weights = model.get_judgment_weights()

    # Also get match/non-match counts for context
    weight_details = []
    for j in sorted(weights.keys(), key=lambda x: -abs(weights[x])):
        w = weights[j]
        mc = model.stats.match_counts.get(j, 0)
        nmc = model.stats.non_match_counts.get(j, 0)
        total = model.stats.total_matches + model.stats.total_non_matches
        weight_details.append({
            "judgment": j,
            "weight": round(w, 4),
            "match_count": mc,
            "non_match_count": nmc,
            "match_rate": round(mc / max(model.stats.total_matches, 1), 4),
            "non_match_rate": round(nmc / max(model.stats.total_non_matches, 1), 4),
        })

    print(f"\nJudgment weights (sorted by |weight|):")
    print(f"{'Judgment':<30s} {'Weight':>8s}  {'P(j|M)':>8s}  {'P(j|¬M)':>8s}")
    print("-" * 60)
    for d in weight_details:
        print(f"{d['judgment']:<30s} {d['weight']:>+8.4f}  {d['match_rate']:>8.4f}  {d['non_match_rate']:>8.4f}")

    # Save weights
    weights_path = ARTIFACTS_DIR / "nars_judgment_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weight_details, f, indent=2)
    print(f"\nSaved weights to {weights_path}")

    # --- Find example pairs ---
    print("\nFinding example pairs...")
    base_rate = model.stats.total_matches / (model.stats.total_matches + model.stats.total_non_matches + 1e-10)
    prior_log_odds = np.log(base_rate / (1 - base_rate + 1e-10))

    tp_example = None
    tn_example = None
    fn_example = None
    fp_example = None

    for id1, id2, label in test_pairs_list:
        rec1 = records_dict.get(id1, {})
        rec2 = records_dict.get(id2, {})
        if not rec1 or not rec2:
            continue

        judgments = model._preprocess(rec1, rec2)
        score = model.stats.compute_score(judgments)
        predicted = 1 if score >= model.threshold else 0

        # Compute per-judgment breakdown
        breakdown = []
        for j in sorted(judgments):
            w = model.stats.get_judgment_weight(j)
            breakdown.append({"judgment": j, "weight": round(w, 4)})

        log_odds_sum = sum(d["weight"] for d in breakdown)

        example = {
            "id1": int(id1),
            "id2": int(id2),
            "label": label,
            "predicted": predicted,
            "score": round(score, 4),
            "threshold": round(model.threshold, 4),
            "judgments": breakdown,
            "log_odds_sum": round(log_odds_sum, 4),
            "prior_log_odds": round(prior_log_odds, 4),
            "total_log_odds": round(log_odds_sum + prior_log_odds, 4),
            "rec1_name": rec1.get("nafn_norm", ""),
            "rec2_name": rec2.get("nafn_norm", ""),
            "rec1_birthyear": rec1.get("birthyear", ""),
            "rec2_birthyear": rec2.get("birthyear", ""),
            "rec1_heimild": rec1.get("heimild", ""),
            "rec2_heimild": rec2.get("heimild", ""),
            "rec1_sex": rec1.get("sex", ""),
            "rec2_sex": rec2.get("sex", ""),
            "rec1_farm": rec1.get("farm", ""),
            "rec2_farm": rec2.get("farm", ""),
        }

        # Look for good examples: clear TP, clear TN, and an interesting error
        if label == 1 and predicted == 1 and score > 0.95 and tp_example is None:
            # Want a TP with several matching fields
            if len(judgments) >= 6:
                tp_example = example
                print(f"  TP: {rec1.get('nafn_norm','')} vs {rec2.get('nafn_norm','')} "
                      f"score={score:.4f}")

        if label == 0 and predicted == 0 and score < 0.1 and tn_example is None:
            if len(judgments) >= 6:
                tn_example = example
                print(f"  TN: {rec1.get('nafn_norm','')} vs {rec2.get('nafn_norm','')} "
                      f"score={score:.4f}")

        # False negative: true match missed
        if label == 1 and predicted == 0 and fn_example is None:
            fn_example = example
            print(f"  FN: {rec1.get('nafn_norm','')} vs {rec2.get('nafn_norm','')} "
                  f"score={score:.4f}")

        # False positive: non-match predicted as match
        if label == 0 and predicted == 1 and fp_example is None:
            if len(judgments) >= 6:
                fp_example = example
                print(f"  FP: {rec1.get('nafn_norm','')} vs {rec2.get('nafn_norm','')} "
                      f"score={score:.4f}")

        if tp_example and tn_example and fn_example and fp_example:
            break

    examples = {
        "true_positive": tp_example,
        "true_negative": tn_example,
        "false_negative": fn_example,
        "false_positive": fp_example,
        "model_info": {
            "threshold": round(model.threshold, 4),
            "base_rate": round(base_rate, 4),
            "prior_log_odds": round(prior_log_odds, 4),
            "total_matches": model.stats.total_matches,
            "total_non_matches": model.stats.total_non_matches,
            "n_train_pairs": len(train_df),
            "n_test_pairs": len(test_pairs_list),
        }
    }

    examples_path = ARTIFACTS_DIR / "nars_pair_explanations.json"
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2, default=str)
    print(f"\nSaved pair explanations to {examples_path}")

    # --- Generate waterfall figure ---
    print("\nGenerating waterfall figure...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use the TP example for the figure
        ex = tp_example
        if ex is None:
            print("  No TP example found, skipping figure")
            return

        judgments = ex["judgments"]
        labels_list = [d["judgment"].replace("_", " ") for d in judgments]
        values = [d["weight"] for d in judgments]

        # Add prior as last bar
        labels_list.append("base-rate prior")
        values.append(ex["prior_log_odds"])

        # Sort by value (positive first, then negative)
        combined = list(zip(labels_list, values))
        combined.sort(key=lambda x: -x[1])
        labels_list = [c[0] for c in combined]
        values = [c[1] for c in combined]

        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        y_pos = range(len(labels_list))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_list, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("LLR contribution to log-odds", fontsize=9)
        ax.axvline(x=0, color="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            x = bar.get_width()
            offset = 0.05 if x >= 0 else -0.05
            ha = "left" if x >= 0 else "right"
            ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}", va="center", ha=ha, fontsize=7)

        total = sum(values)
        score = 1.0 / (1.0 + np.exp(-total))
        ax.set_title(
            f"True match: score = $\\sigma$({total:+.2f}) = {score:.3f}",
            fontsize=9, pad=8
        )

        plt.tight_layout()

        fig_path = FIGURES_DIR / "nars_pair_explanation.pdf"
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        fig.savefig(str(fig_path).replace(".pdf", ".png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved figure to {fig_path}")

    except ImportError as e:
        print(f"  matplotlib not available: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
