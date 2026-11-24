"""Logistic regression over ONA's own judgments: the learned combiner of the same evidence.

The like-for-like control for the gap between NARS (ONA) and the learned matchers. It sees exactly
what ONA and the additive control see -- the binary indicators of the pair's judgment set, no
middle name, marital status or continuous feature -- and is trained on the same pairs, with the
paper's logistic-regression settings (fitted_models.fit_model: StandardScaler, L2, C = 1,
2,000 iterations, random_state = seed) and the same decision rule as both engines
(run_experiments.threshold_on_validation on the selection role, highest tied observed score).

* Controlled pairs: fit on the seed's 12,000 controlled training pairs (judgment sets from
  pairs/seed_<s>/controlled.csv.gz), threshold on the controlled selection pairs, score the
  controlled test draw and the second assessment's controlled pairs.
* Candidate graphs, controlled-pair training: the same model on the controlled-trained graph
  (publication_graph), threshold on its selection role.
* Candidate graphs, candidate-matched training: fit on the seed's candidate-matched training
  pairs (candidate-quarantined-extended/seed_<s>/train_pairs.csv). Their judgment sets are
  recomputed with the saved candidate count model's own preprocessing and must reproduce its
  training counts exactly. Threshold on the candidate-graph selection role; score the test graph
  and the second assessment's graph. Partitions use publication_evaluation.full_node_clusters.

Writes experiments/ona/full/iceid/judgment_lr/ (per-pair test scores, fit.json) and
experiments/ona/full/iceid/tables/judgment_lr_metrics.csv. It reads only saved pair files, the record cache of
experiments/ona/full/iceid/inputs, the saved candidate count models and the graph node lists.
Usage (from label-repair): python run_judgment_lr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "vendor/bench"))

from run_experiments import metrics, threshold_on_validation  # noqa: E402
from publication_evaluation import full_node_clusters, tie_averaged_precision_at_positive_count  # noqa: E402
from run_ona_full import ICE, PUB, CAND, LOCKED, SEEDS, dump, load_records  # noqa: E402

OUT = ICE / "judgment_lr"
MODEL = "lr_judgments"


def pairs(seed, stem):
    return pd.read_csv(ICE / f"pairs/seed_{seed}/{stem}.csv.gz", float_precision="round_trip")


def part(frame, partition):
    return frame[frame.partition.eq(partition)].reset_index(drop=True)


def nodes(path):
    return pd.read_csv(path).set_index("id", drop=False)


def vocabulary():
    sets = pd.read_csv(ICE / "judgment_sets.csv.gz").set_index("judgment_set").judgments.str.split()
    words = sorted({j for v in sets for j in v})
    return {k: frozenset(v) for k, v in sets.items()}, words


def design(judgment_sets, words):
    """Binary judgment indicators, one column per judgment of the vocabulary (sorted)."""
    index = {j: i for i, j in enumerate(words)}
    matrix = np.zeros((len(judgment_sets), len(words)), dtype=float)
    for row, judgments in enumerate(judgment_sets):
        for j in judgments:
            matrix[row, index[j]] = 1.0
    return matrix


def fit(seed, x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))  # fitted_models.fit_model
    model.fit(x, y)
    return model


def score(model, sets, words, frame):
    return model.predict_proba(design([sets[s] for s in frame.judgment_set], words))[:, 1]


def candidate_training_sets(seed, records):
    """Judgment sets of the candidate-matched training pairs, checked against the saved count model."""
    import joblib
    saved = joblib.load(CAND / f"seed_{seed}/published_nars.joblib")
    train = pd.read_csv(CAND / f"seed_{seed}/train_pairs.csv")
    judgments = [frozenset(saved._preprocess(records[int(a)], records[int(b)])) for a, b in zip(train.id1, train.id2)]
    match, non_match = {}, {}
    for js, label in zip(judgments, train.label):
        for j in js:
            target = match if label else non_match
            target[j] = target.get(j, 0) + 1
    stats = saved.stats
    keys = set(match) | set(non_match) | set(stats.match_counts) | set(stats.non_match_counts)
    if (stats.total_matches != int(train.label.sum()) or stats.total_non_matches != int((1 - train.label).sum())
            or any(match.get(k, 0) != stats.match_counts.get(k, 0) or non_match.get(k, 0) != stats.non_match_counts.get(k, 0)
                   for k in keys)):
        raise RuntimeError(f"seed {seed}: recomputed candidate-training judgments differ from the saved counts")
    return train, judgments


def controlled_training_check(seed, sets, train):
    """The controlled training judgment sets must reproduce ONA's 'pub' implication counts."""
    knowledge = pd.read_csv(ICE / "tables/knowledge.csv")
    k = knowledge[knowledge.seed.eq(seed) & knowledge.knowledge_base.eq("pub")].set_index("judgment")
    for judgment, row in k.iterrows():
        has = train.judgment_set.map(lambda s: judgment in sets[s])
        if int(has.sum()) != int(row.w) or int((has & train.label.eq(1)).sum()) != int(row.w_plus):
            raise RuntimeError(f"seed {seed}: controlled training counts differ for {judgment}")


def graph_row(seed, population, training, labels, values, threshold, node_frame, frame):
    cluster, clusters = full_node_clusters(node_frame, frame, values, threshold)
    row = {"seed": seed, "model": MODEL, "population": population, "training": training,
           **metrics(labels, values, threshold), **cluster,
           "p_at_k": tie_averaged_precision_at_positive_count(labels, values)}
    return row, clusters


def main():
    from resource_limits import check_memory
    check_memory(2)
    sets, words = vocabulary()
    records = load_records()
    locked_nodes = nodes(LOCKED / "candidate_nodes.csv")
    rows, report = [], {"model": "StandardScaler + LogisticRegression(C=1, L2, max_iter=2000, random_state=seed)",
                        "features": words, "threshold_rule": "run_experiments.threshold_on_validation (selection role)",
                        "seeds": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        controlled = pairs(seed, "controlled")
        train, selection, test = (part(controlled, p) for p in ["train", "selection", "test"])
        controlled_training_check(seed, sets, train)
        model = fit(seed, design([sets[s] for s in train.judgment_set], words), train.label.to_numpy())
        threshold = threshold_on_validation(selection.label.to_numpy(), score(model, sets, words, selection))
        test_scores = score(model, sets, words, test)
        rows.append({"seed": seed, "model": MODEL, "population": "controlled", "training": "controlled",
                     **metrics(test.label.to_numpy(), test_scores, threshold)})
        locked_c = pairs(seed, "locked_controlled")
        locked_c_scores = score(model, sets, words, locked_c)
        rows.append({"seed": seed, "model": MODEL, "population": "locked_controlled", "training": "controlled",
                     **metrics(locked_c.label.to_numpy(), locked_c_scores, threshold)})
        # Controlled-pair training applied to the candidate graph (as ONA's graph_metrics.csv rows).
        graph = pairs(seed, "publication_graph")
        g_sel, g_test = part(graph, "selection"), part(graph, "test")
        g_threshold = threshold_on_validation(g_sel.label.to_numpy(), score(model, sets, words, g_sel))
        test_nodes = nodes(PUB / f"seed_{seed}/test_graph_nodes.csv")
        row, _ = graph_row(seed, "candidate_graph", "controlled", g_test.label.to_numpy(),
                           score(model, sets, words, g_test), g_threshold, test_nodes, g_test)
        rows.append(row)
        coefficients = dict(zip(words, (model[-1].coef_[0] / model[0].scale_).tolist()))
        # Candidate-matched training.
        cand_train, cand_sets = candidate_training_sets(seed, records)
        cand_model = fit(seed, design(cand_sets, words), cand_train.label.to_numpy())
        cgraph = pairs(seed, "candidate_graph")
        c_sel, c_test = part(cgraph, "selection"), part(cgraph, "test")
        if not (c_test[["id1", "id2", "label"]].equals(g_test[["id1", "id2", "label"]])):
            raise RuntimeError("Candidate and publication graphs differ in test pairs")
        c_threshold = threshold_on_validation(c_sel.label.to_numpy(), score(cand_model, sets, words, c_sel))
        c_scores = score(cand_model, sets, words, c_test)
        row, clusters = graph_row(seed, "candidate_graph", "candidate_matched", c_test.label.to_numpy(), c_scores,
                                  c_threshold, test_nodes, c_test)
        rows.append(row)
        locked_g = pairs(seed, "locked_candidate")
        lg_scores = score(cand_model, sets, words, locked_g)
        row, _ = graph_row(seed, "locked_candidate", "candidate_matched", locked_g.label.to_numpy(), lg_scores,
                           c_threshold, locked_nodes, locked_g)
        rows.append(row)
        folder = OUT / f"seed_{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame, values in [("controlled_test", test, test_scores), ("locked_controlled", locked_c, locked_c_scores),
                                    ("candidate_test", c_test, c_scores), ("locked_candidate", locked_g, lg_scores)]:
            frame[["id1", "id2", "label"]].assign(score=values).to_csv(folder / f"{name}.csv.gz", index=False)
        clusters.to_csv(folder / "candidate_test_clusters.csv.gz", index=False)
        report["seeds"][str(seed)] = {
            "controlled_threshold": threshold, "controlled_graph_threshold": g_threshold, "candidate_threshold": c_threshold,
            "controlled_training_pairs": len(train), "candidate_training_pairs": len(cand_train),
            "candidate_training_judgment_sets": len(set(cand_sets)),
            "coefficients_per_unit_indicator_controlled": coefficients,
            "coefficients_per_unit_indicator_candidate": dict(zip(words, (cand_model[-1].coef_[0] / cand_model[0].scale_).tolist())),
            "candidate_training_counts_equal_saved_model": True, "controlled_training_counts_equal_ona_knowledge": True}
        print(f"seed {seed}: controlled F1 {rows[-5]['f1']:.4f}, candidate graph F1 {rows[-2]['f1']:.4f} ARI {rows[-2]['ari']:.4f}",
              flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ICE / "tables/judgment_lr_metrics.csv", index=False)
    dump(OUT / "fit.json", report)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
