"""Three-seed engine comparison, ICE-ID tie-rule sensitivity and score-distribution facts.

Re-analysis of the saved per-pair score files only (no refitting, no ONA run):

* Controlled-pair component bootstrap for all three seeds (the seed-131 analysis in
  run_ona_metrics.bootstraps is reproduced first with the same rng and resamples). Each seed's
  test draw has its own observed identity components, resampled independently; the three-seed
  mean F1 difference is recomputed on every joint resample. The second assessment uses one case
  set for every seed, so one weight vector per resample is shared across seeds.
* Candidate graphs are not bootstrapped: their evaluated pairs form one connected identity
  component, so the component bootstrap has no independent units. Per-seed values are reported.
* ICE-ID tie rule: threshold_on_validation keeps the highest of the selection-F1-tied thresholds;
  the lowest tied threshold is applied to the same test scores (controlled pairs, candidate
  graph, second assessment) and the effect on F1 and ARI is recorded.
* Score facts: distinct scores per test set and engine (ICE-ID and external), share of reference
  matches whose ONA expectation is below 0.5, largest expectation, and answer confidence by the
  number of judgments in an episode.

* Reliability bins (raw, sigmoid, isotonic; test pairs pooled over seeds) and the F1 change of both
  engines after the external label audit's removals (the audit's own rules and removal sets), under
  the external grid rule and under the ICE-ID rule.
* Engine interval with each engine's threshold reselected on every resampled selection set.
* External benchmarks: both threshold rules (the 181-point grid of results.csv and the ICE-ID rule,
  highest tied observed validation score), the best F1 of any judgment-set rule (test-label oracle),
  and a paired pair-level bootstrap of ONA minus control with reselected thresholds.
* Decision disagreements between the engines; candidate-graph recall for true matches that moved
  (commune and county codes both differ, recodings excluded; see Geography) versus all other true
  matches, for every graph scorer, and the movers' share of training matches; clusters holding two
  records of one census wave; ONA with a veto on the birth-year judgment the label screens make a
  perfect non-match signal. Logistic regression over ONA's judgments (run_judgment_lr.py) joins the
  controlled-pair bootstraps and the graph diagnostics.

Writes experiments/ona/full/iceid/tables/engine_comparison.json, ice_tie_sensitivity.csv,
reliability_bins.csv, label_audit_engines.csv, external_threshold_rules.csv, engine_disagreements.csv,
graph_movers.csv, graph_wave_repeats.csv, mover_training_shares.csv, birthyear_veto.csv and
setcount_graph_movers.csv (judgment-set ONA, experiments/ona/setcount, under the same mover rule).
Usage (from label-repair): python run_ona_engine_comparison.py
       python run_ona_engine_comparison.py --judgment-diagnostics   (the last five files only)
       python run_ona_engine_comparison.py --setcount-movers        (setcount_graph_movers.csv only)
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
from publication_evaluation import observed_pair_components, full_node_clusters  # noqa: E402
from run_ona_full import ICE, OUT, SEEDS, PUB, LOCKED, dump, load_persons, load_records  # noqa: E402

TABLES = ICE / "tables"
NEURAL = BASE / "experiments/neural-quarantined-extended"
ENGINES = {"ona": "ona_score", "published_nars": "llr_score"}
CLASSICAL = ["fellegi_sunter", "rules", "logistic_regression", "gradient_boosting", "mlp"]
NEURAL_MODELS = ["ditto", "hiergat"]
JUDGMENT_LR = "lr_judgments"  # logistic regression over ONA's judgments (run_judgment_lr.py)
JUDGMENT_LR_DIR = ICE / "judgment_lr"
CAND = BASE / "experiments/candidate-quarantined-extended"
REPEATS = 2000


def pairs(seed, stem):
    return pd.read_csv(ICE / f"pairs/seed_{seed}/{stem}.csv.gz", float_precision="round_trip")


def part(frame, partition):
    return frame[frame.partition.eq(partition)].reset_index(drop=True)


def weighted_f1(labels, decision, weights):
    """F1 with integer case weights; equals sklearn f1_score(sample_weight=..., zero_division=0)."""
    tp = np.sum(weights * (decision & labels))
    fp = np.sum(weights * (decision & ~labels))
    fn = np.sum(weights * (~decision & labels))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def low_tie_threshold(labels, scores):
    """threshold_on_validation with the lowest instead of the highest tied threshold."""
    from sklearn.metrics import precision_recall_curve
    p, r, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    tied = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    return float(thresholds[tied[0]]), int(len(tied))


# ------------------------------------------------------------------ controlled-pair bootstrap
def controlled_cases(seed):
    """Seed's controlled test cases with every engine's and comparator's scores and thresholds."""
    controlled = pairs(seed, "controlled")
    test, selection = part(controlled, "test"), part(controlled, "selection")
    vd = PUB / f"seed_{seed}/corrected_training_labels"
    saved = pd.read_csv(vd / "test_predictions.csv", float_precision="round_trip")
    if not saved[["id1", "id2", "label"]].equals(test[["id1", "id2", "label"]]):
        raise RuntimeError(f"seed {seed}: saved test predictions differ in cases")
    scores, thresholds = {}, {}
    for engine, column in ENGINES.items():
        scores[engine] = test[column].to_numpy(float)
        thresholds[engine] = threshold_on_validation(selection.label.to_numpy(), selection[column].to_numpy(float))
    for name in CLASSICAL:
        scores[name] = saved[name].to_numpy(float)
        thresholds[name] = json.loads((vd / f"{name}_decisions.json").read_text())["controlled_threshold"]
    for name in NEURAL_MODELS:
        neural = pd.read_csv(NEURAL / f"seed_{seed}/{name}/test_predictions.csv", float_precision="round_trip")
        merged = test[["id1", "id2", "label"]].merge(neural, on=["id1", "id2"], how="left", suffixes=("", "_n"),
                                                    validate="one_to_one")
        if merged.score.isna().any() or not (merged.label == merged.label_n).all():
            raise RuntimeError(f"seed {seed}: {name} predictions do not cover the test cases")
        scores[name] = merged.score.to_numpy(float)
        thresholds[name] = json.loads((NEURAL / f"seed_{seed}/{name}/decisions.json").read_text())["threshold"]
    scores[JUDGMENT_LR], thresholds[JUDGMENT_LR] = judgment_lr(seed, "controlled_test", test)
    return test, scores, thresholds


def judgment_lr(seed, stem, cases):
    """Scores and threshold of logistic regression over ONA's judgments (run_judgment_lr.py)."""
    saved = pd.read_csv(JUDGMENT_LR_DIR / f"seed_{seed}/{stem}.csv.gz", float_precision="round_trip")
    if not saved[["id1", "id2", "label"]].equals(cases[["id1", "id2", "label"]].reset_index(drop=True)):
        raise RuntimeError(f"seed {seed}: judgment-LR {stem} scores differ in cases")
    fit = json.loads((JUDGMENT_LR_DIR / "fit.json").read_text())["seeds"][str(seed)]
    key = "candidate_threshold" if stem.startswith(("candidate", "locked_candidate")) else "controlled_threshold"
    return saved.score.to_numpy(float), float(fit[key])


def check_seed131(persons, test, scores, thresholds):
    """The saved seed-131 ONA-minus-control interval must be reproduced (rng 571, 500 resamples)."""
    saved = json.loads((TABLES / "uncertainty.json").read_text())
    expected = saved["publication_test"]["standard_reference_llr"]["models"]["ona"]["paired_delta_vs_llr_interval"]
    inverse = observed_pair_components(test, persons)
    n = int(inverse.max()) + 1
    rng = np.random.default_rng(571)
    labels = test.label.to_numpy().astype(bool)
    d = {e: scores[e] >= thresholds[e] for e in ENGINES}
    deltas = []
    for _ in range(500):
        w = np.bincount(rng.integers(0, n, n), minlength=n)[inverse]
        deltas.append(weighted_f1(labels, d["ona"], w) - weighted_f1(labels, d["published_nars"], w))
    got = np.quantile(deltas, [0.025, 0.975])
    if not np.allclose(got, expected, rtol=0, atol=1e-12):
        raise RuntimeError(f"Seed-131 bootstrap not reproduced: {got} vs {expected}")
    return got.tolist()


def bootstrap_main(persons):
    """Independent component resamples per seed; three-seed mean of model-minus-reference F1."""
    seeds = {}
    for seed in SEEDS:
        test, scores, thresholds = controlled_cases(seed)
        labels = test.label.to_numpy().astype(bool)
        inverse = observed_pair_components(test, persons)
        seeds[seed] = {"labels": labels, "inverse": inverse, "n": int(inverse.max()) + 1,
                       "decisions": {m: scores[m] >= thresholds[m] for m in scores}, "thresholds": thresholds,
                       "observed": {m: weighted_f1(labels, scores[m] >= thresholds[m], np.ones(len(labels))) for m in scores}}
        if seed == SEEDS[0]:
            reproduced = check_seed131(persons, test, scores, thresholds)
    models = list(seeds[SEEDS[0]]["decisions"])
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(571).spawn(len(SEEDS))]
    samples = {seed: {m: np.empty(REPEATS) for m in models} for seed in SEEDS}
    for i in range(REPEATS):
        for rng, seed in zip(rngs, SEEDS):
            s = seeds[seed]
            w = np.bincount(rng.integers(0, s["n"], s["n"]), minlength=s["n"])[s["inverse"]]
            for m in models:
                samples[seed][m][i] = weighted_f1(s["labels"], s["decisions"][m], w)
    return summarise(seeds, samples, models), reproduced


def bootstrap_locked(persons):
    """Second assessment: one case set for all seeds, so each resample's weights are shared across seeds."""
    cases = pd.read_csv(LOCKED / "controlled_pairs.csv")
    labels = cases.label.to_numpy().astype(bool)
    inverse = observed_pair_components(cases, persons)
    n = int(inverse.max()) + 1
    seeds = {}
    for seed in SEEDS:
        frame = pairs(seed, "locked_controlled")
        if not frame[["id1", "id2", "label"]].equals(cases[["id1", "id2", "label"]]):
            raise RuntimeError("Locked ONA pairs differ from the locked cases")
        controlled = pairs(seed, "controlled")
        selection = part(controlled, "selection")
        decisions = {e: frame[c].to_numpy(float) >= threshold_on_validation(selection.label.to_numpy(),
                                                                             selection[c].to_numpy(float))
                     for e, c in ENGINES.items()}
        vd = PUB / f"seed_{seed}/corrected_training_labels"
        for name in CLASSICAL + NEURAL_MODELS:
            saved = pd.read_csv(LOCKED / f"seed_{seed}/controlled_{name}_predictions.csv.gz", float_precision="round_trip")
            if not saved[["id1", "id2", "label"]].equals(cases[["id1", "id2", "label"]]):
                raise RuntimeError("Locked predictions differ in cases")
            threshold = (json.loads((NEURAL / f"seed_{seed}/{name}/decisions.json").read_text())["threshold"]
                         if name in NEURAL_MODELS else json.loads((vd / f"{name}_decisions.json").read_text())["controlled_threshold"])
            decisions[name] = saved.score.to_numpy(float) >= threshold
        values, threshold = judgment_lr(seed, "locked_controlled", cases)
        decisions[JUDGMENT_LR] = values >= threshold
        seeds[seed] = {"labels": labels, "decisions": decisions,
                       "observed": {m: weighted_f1(labels, d, np.ones(len(labels))) for m, d in decisions.items()}}
    models = list(seeds[SEEDS[0]]["decisions"])
    rng = np.random.default_rng(np.random.SeedSequence(572))
    samples = {seed: {m: np.empty(REPEATS) for m in models} for seed in SEEDS}
    for i in range(REPEATS):
        w = np.bincount(rng.integers(0, n, n), minlength=n)[inverse]
        for seed in SEEDS:
            for m in models:
                samples[seed][m][i] = weighted_f1(labels, seeds[seed]["decisions"][m], w)
    out = summarise(seeds, samples, models)
    out["n_components"] = n
    return out


def summarise(seeds, samples, models):
    q = lambda v: np.quantile(v, [0.025, 0.975]).tolist()
    result = {"repeats": REPEATS, "per_seed_f1": {}, "versus_control": {}, "versus_ona": {}}
    for m in models:
        result["per_seed_f1"][m] = {str(s): seeds[s]["observed"][m] for s in SEEDS}
    for reference, key in [("published_nars", "versus_control"), ("ona", "versus_ona")]:
        for m in models:
            if m == reference:
                continue
            per_seed = {str(s): {"observed_delta": seeds[s]["observed"][m] - seeds[s]["observed"][reference],
                                 "interval": q(samples[s][m] - samples[s][reference])} for s in SEEDS}
            mean = np.mean([samples[s][m] - samples[s][reference] for s in SEEDS], axis=0)
            result[key][m] = {"per_seed": per_seed,
                              "mean_delta": float(np.mean([per_seed[str(s)]["observed_delta"] for s in SEEDS])),
                              "mean_delta_interval": q(mean)}
    if "n" in seeds[SEEDS[0]]:
        result["n_components"] = {str(s): seeds[s]["n"] for s in SEEDS}
    return result


# ------------------------------------------------------------------ ICE-ID tie rule
def tie_sensitivity():
    rows = []
    locked_nodes = pd.read_csv(LOCKED / "candidate_nodes.csv").set_index("id", drop=False)
    for seed in SEEDS:
        controlled = pairs(seed, "controlled")
        graph = pairs(seed, "candidate_graph")
        locked_c, locked_g = pairs(seed, "locked_controlled"), pairs(seed, "locked_candidate")
        graph_nodes = pd.read_csv(PUB / f"seed_{seed}/test_graph_nodes.csv").set_index("id", drop=False)
        for engine, column in ENGINES.items():
            for population, selection, tests in [
                    ("controlled", part(controlled, "selection"),
                     [("test", part(controlled, "test"), None), ("locked", locked_c, None)]),
                    ("candidate", part(graph, "selection"),
                     [("test", part(graph, "test"), graph_nodes), ("locked", locked_g, locked_nodes)])]:
                y, s = selection.label.to_numpy(), selection[column].to_numpy(float)
                high = threshold_on_validation(y, s)
                low, n_tied = low_tie_threshold(y, s)
                for role, frame, node_frame in tests:
                    ty, ts = frame.label.to_numpy(), frame[column].to_numpy(float)
                    row = {"seed": seed, "model": engine, "population": population, "role": role,
                           "tied_thresholds": n_tied, "threshold_high": high, "threshold_low": low}
                    for name, t in [("high", high), ("low", low)]:
                        m = metrics(ty, ts, t)
                        row.update({f"f1_{name}": m["f1"], f"fp_{name}": m["fp"], f"fn_{name}": m["fn"]})
                    if node_frame is not None:
                        row["ari_high"] = full_node_clusters(node_frame, frame, ts, high)[0]["ari"]
                        row["ari_low"] = (row["ari_high"] if low == high else
                                          full_node_clusters(node_frame, frame, ts, low)[0]["ari"])
                    rows.append(row)
    frame = pd.DataFrame(rows)
    # The highest-tie rows must reproduce the saved tables.
    saved = {("controlled", "test"): pd.read_csv(TABLES / "metrics.csv"),
             ("candidate", "test"): pd.read_csv(TABLES / "candidate_graph_metrics.csv"),
             ("controlled", "locked"): pd.read_csv(TABLES / "locked_metrics.csv"),
             ("candidate", "locked"): pd.read_csv(TABLES / "locked_metrics.csv")}
    for r in frame.itertuples():
        s = saved[(r.population, r.role)]
        s = s[s.model.eq(r.model) & s.seed.eq(r.seed)]
        if "evaluation" in s:
            s = s[s.evaluation.eq("controlled_crosswave_reference")]
        if r.role == "locked":
            s = s[s.population.eq(r.population)]
        if len(s) != 1 or abs(float(s.f1.iloc[0]) - r.f1_high) > 1e-12:
            raise RuntimeError(f"Tie rows do not reproduce the saved F1: {r}")
        if r.population == "candidate" and abs(float(s.ari.iloc[0]) - r.ari_high) > 1e-12:
            raise RuntimeError(f"Tie rows do not reproduce the saved ARI: {r}")
    return frame


# ------------------------------------------------------------------ score facts
def score_facts():
    facts = {"iceid_controlled_test": {}, "external_test": {}}
    for seed in SEEDS:
        test = part(pairs(seed, "controlled"), "test")
        positive = test[test.label.eq(1)]
        facts["iceid_controlled_test"][str(seed)] = {
            "pairs": len(test), "distinct_ona": int(test.ona_score.nunique()), "distinct_control": int(test.llr_score.nunique()),
            "ona_min": float(test.ona_score.min()), "ona_max": float(test.ona_score.max()),
            "positives_below_half": int((positive.ona_score < 0.5).sum()), "positives": len(positive),
            "positive_share_below_half": float((positive.ona_score < 0.5).mean()),
            "positive_median": float(positive.ona_score.median())}
    for folder in sorted((OUT / "external").iterdir()):
        if folder.is_dir():
            t = pd.read_csv(folder / "test_scores.csv.gz")
            facts["external_test"][folder.name] = {"pairs": len(t), "distinct_ona": int(t.ona_score.nunique()),
                                                   "distinct_control": int(t.llr_score.nunique())}
    episodes = pd.read_csv(OUT / "episodes/iceid/episodes.csv.gz", usecols=["n_judgments", "confidence"])
    by = episodes.groupby("n_judgments").confidence.agg(["size", "median"])
    facts["answer_confidence_by_judgments"] = {str(int(k)): {"episodes": int(v["size"]), "median_confidence": float(v["median"])}
                                               for k, v in by.iterrows()}
    from scipy.stats import spearmanr
    facts["answer_confidence_spearman"] = float(spearmanr(episodes.n_judgments, episodes.confidence).statistic)
    return facts


# ------------------------------------------------------------------ reliability bins
def reliability(bins=10):
    """Observed match rate against mean score in score-quantile bins, test pairs pooled over seeds.

    Maps are fitted on each seed's calibration partition (publication_run.fit_calibrators), as in
    run_ona_metrics; controlled pairs use controlled training, candidate graphs candidate-matched.
    """
    from publication_run import fit_calibrators, calibrated_scores
    rows = []
    for population, stem in [("controlled", "controlled"), ("candidate", "candidate_graph")]:
        pooled = {}
        for seed in SEEDS:
            frame = pairs(seed, stem)
            cal, test = part(frame, "calibration"), part(frame, "test")
            for engine, column in ENGINES.items():
                maps = fit_calibrators(cal.label.to_numpy(), cal[column].to_numpy(float))
                raw = test[column].to_numpy(float)
                for method, values in [("raw", raw), *[(m, calibrated_scores(raw, c)) for m, c in maps.items()]]:
                    pooled.setdefault((engine, method), []).append(pd.DataFrame({"score": values, "label": test.label}))
        for (engine, method), parts in pooled.items():
            frame = pd.concat(parts, ignore_index=True)
            edges = np.unique(np.quantile(frame.score, np.linspace(0, 1, bins + 1)))
            index = np.clip(np.searchsorted(edges, frame.score, side="right") - 1, 0, len(edges) - 2)
            for b, group in frame.groupby(index):
                rows.append({"population": population, "model": engine, "calibration": method, "bin": int(b),
                             "n": len(group), "mean_score": float(group.score.mean()),
                             "observed_rate": float(group.label.mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ label audit, NARS (ONA) and control
def label_audit_engines():
    """F1 change of both engines after the label audit's removals (audit_labels.main rules)."""
    audit = BASE / "experiments/completion/benchmark_label_audit"
    sys.path.insert(0, str(audit))
    from audit_labels import DEFINITE, f1  # noqa: E402
    flags = pd.read_csv(audit / "flagged_pairs.csv", dtype={"id1": str, "id2": str}, keep_default_na=False)
    results = pd.read_csv(OUT / "external/results.csv")
    rows = []
    for dataset in sorted(results.dataset.unique()):
        test = pd.read_csv(OUT / f"external/{dataset}/test_scores.csv.gz", dtype={"ltable_id": str, "rtable_id": str})
        f = flags[flags.dataset.eq(dataset) & flags.split.eq("test")]
        positions = f.row.astype(int).to_numpy()
        if len(f) and not ((test.ltable_id.to_numpy()[positions] == f.id1.to_numpy()).all()
                           and (test.rtable_id.to_numpy()[positions] == f.id2.to_numpy()).all()):
            raise RuntimeError(f"{dataset}: audit rows do not address the ONA test pairs")
        drop1 = set(f[f.check.isin(DEFINITE)].row.astype(int))
        drop2 = drop1 | set(f[f.check.isin(["c_fn", "c_fp"]) & f.verdict.eq("label_error")].row.astype(int))
        keep1 = ~np.isin(np.arange(len(test)), list(drop1))
        keep2 = ~np.isin(np.arange(len(test)), list(drop2))
        gold = test.label.to_numpy()
        valid = pd.read_csv(OUT / f"external/{dataset}/valid_scores.csv.gz")
        for engine, column in ENGINES.items():
            r = results[results.dataset.eq(dataset) & results.model.eq(engine)].iloc[0]
            pred = (test[column].to_numpy(float) >= r.threshold).astype(int)
            base = f1(gold, pred)
            if abs(base - r.f1) > 5e-5:
                raise RuntimeError(f"{dataset}/{engine}: F1 does not reproduce results.csv")
            # The reported rule (as on ICE-ID): highest tied observed validation score.
            observed = (test[column].to_numpy(float) >= threshold_on_validation(
                valid.label.to_numpy(), valid[column].to_numpy(float))).astype(int)
            rows.append({"dataset": dataset, "model": engine, "n_test": len(test), "f1": base,
                         "n_drop_definite": int((~keep1).sum()), "f1_drop_definite": f1(gold[keep1], pred[keep1]),
                         "n_drop_definite_and_c": int((~keep2).sum()), "f1_drop_definite_and_c": f1(gold[keep2], pred[keep2]),
                         "f1_observed": f1(gold, observed), "f1_observed_drop_definite": f1(gold[keep1], observed[keep1]),
                         "f1_observed_drop_definite_and_c": f1(gold[keep2], observed[keep2])})
    frame = pd.DataFrame(rows)
    frame["delta_definite"] = frame.f1_drop_definite - frame.f1
    frame["delta_definite_and_c"] = frame.f1_drop_definite_and_c - frame.f1
    frame["delta_observed_definite"] = frame.f1_observed_drop_definite - frame.f1_observed
    frame["delta_observed_definite_and_c"] = frame.f1_observed_drop_definite_and_c - frame.f1_observed
    sensitivity = pd.read_csv(audit / "f1_sensitivity.csv")
    first = sensitivity.drop_duplicates("dataset").set_index("dataset")
    for r in frame.itertuples():  # same removals as the audit's own table
        if (r.n_drop_definite, r.n_drop_definite_and_c) != (first.loc[r.dataset, "n_drop_definite"],
                                                            first.loc[r.dataset, "n_drop_definite_and_c"]):
            raise RuntimeError(f"{r.dataset}: removal counts differ from f1_sensitivity.csv")
    return frame


# ------------------------------------------------------------------ thresholds reselected inside the bootstrap
def weighted_threshold(labels, scores, weights):
    """threshold_on_validation with integer case weights: F1-maximising observed score, highest tie.

    Candidate thresholds are the scores of cases with positive weight (the resampled selection set);
    with unit weights this equals run_experiments.threshold_on_validation (checked in the callers).
    """
    present = weights > 0
    values = np.unique(scores[present])
    order = np.argsort(-scores, kind="stable")
    s, y, w = scores[order], labels[order].astype(bool), weights[order]
    tp_cum, fp_cum = np.cumsum(w * y), np.cumsum(w * ~y)
    positives = tp_cum[-1]
    # Cases with score >= t: the last index of s (descending) whose value >= t.
    last = np.searchsorted(-s, -values, side="right") - 1
    tp, fp = tp_cum[last], fp_cum[last]
    f1 = np.where(2 * tp + fp + (positives - tp) > 0, 2 * tp / np.maximum(2 * tp + fp + positives - tp, 1e-300), 0.0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(values[best])


def bootstrap_reselected(persons):
    """ONA minus control with each engine's threshold reselected on the resampled selection pairs.

    Main draws: per seed, the selection and test pairs' identity components are resampled
    independently and the cutoff is reselected by the paper's rule on every replicate. Second
    assessment: one test resample shared by the seeds (one case set), selection resampled per seed.
    """
    rng = np.random.default_rng(np.random.SeedSequence(573))
    prepared, locked_cases = {}, pd.read_csv(LOCKED / "controlled_pairs.csv")
    locked_inverse = observed_pair_components(locked_cases, persons)
    for seed in SEEDS:
        controlled = pairs(seed, "controlled")
        selection, test = part(controlled, "selection"), part(controlled, "test")
        locked = pairs(seed, "locked_controlled")
        if not locked[["id1", "id2", "label"]].equals(locked_cases[["id1", "id2", "label"]]):
            raise RuntimeError("Locked ONA pairs differ from the locked cases")
        entry = {"sel_y": selection.label.to_numpy().astype(bool), "test_y": test.label.to_numpy().astype(bool),
                 "sel_inverse": observed_pair_components(selection, persons), "test_inverse": observed_pair_components(test, persons),
                 "sel": {}, "test": {}, "locked": {}}
        for engine, column in ENGINES.items():
            entry["sel"][engine] = selection[column].to_numpy(float)
            entry["test"][engine] = test[column].to_numpy(float)
            entry["locked"][engine] = locked[column].to_numpy(float)
            ones = np.ones(len(selection), dtype=np.int64)
            if weighted_threshold(entry["sel_y"], entry["sel"][engine], ones) != threshold_on_validation(
                    entry["sel_y"], entry["sel"][engine]):
                raise RuntimeError("Weighted threshold selection does not reproduce threshold_on_validation")
        prepared[seed] = entry
    locked_y = locked_cases.label.to_numpy().astype(bool)
    n_locked = int(locked_inverse.max()) + 1
    main, locked_delta = np.empty(REPEATS), np.empty(REPEATS)
    observed = {}
    for seed, e in prepared.items():
        t = {g: threshold_on_validation(e["sel_y"], e["sel"][g]) for g in ENGINES}
        ones = np.ones(len(e["test_y"]))
        observed[seed] = {g: weighted_f1(e["test_y"], e["test"][g] >= t[g], ones) for g in ENGINES}
    for i in range(REPEATS):
        lw = np.bincount(rng.integers(0, n_locked, n_locked), minlength=n_locked)[locked_inverse]
        deltas, locked_deltas = [], []
        for seed, e in prepared.items():
            n_sel, n_test = int(e["sel_inverse"].max()) + 1, int(e["test_inverse"].max()) + 1
            sw = np.bincount(rng.integers(0, n_sel, n_sel), minlength=n_sel)[e["sel_inverse"]]
            tw = np.bincount(rng.integers(0, n_test, n_test), minlength=n_test)[e["test_inverse"]]
            f, g = {}, {}
            for engine in ENGINES:
                t = weighted_threshold(e["sel_y"], e["sel"][engine], sw)
                f[engine] = weighted_f1(e["test_y"], e["test"][engine] >= t, tw)
                g[engine] = weighted_f1(locked_y, e["locked"][engine] >= t, lw)
            deltas.append(f["ona"] - f["published_nars"]); locked_deltas.append(g["ona"] - g["published_nars"])
        main[i], locked_delta[i] = np.mean(deltas), np.mean(locked_deltas)
    q = lambda v: np.quantile(v, [0.025, 0.975]).tolist()
    return {"repeats": REPEATS, "rule": "threshold_on_validation reselected on each resampled selection set",
            "observed_mean_delta": float(np.mean([observed[s]["ona"] - observed[s]["published_nars"] for s in SEEDS])),
            "main_interval": q(main), "main_median": float(np.median(main)),
            "locked_interval": q(locked_delta), "locked_median": float(np.median(locked_delta))}


# ------------------------------------------------------------------ external threshold rules, oracle, paired bootstrap
def set_oracle(labels, judgment_sets):
    """Best F1 of any rule that links whole judgment sets: sets ordered by their own (test) match rate."""
    frame = pd.DataFrame({"set": judgment_sets, "y": labels.astype(int)})
    groups = frame.groupby("set").y.agg(["sum", "size"])
    groups = groups.assign(rate=groups["sum"] / groups["size"]).sort_values("rate", ascending=False)
    tp, predicted, positives = groups["sum"].cumsum().to_numpy(), groups["size"].cumsum().to_numpy(), frame.y.sum()
    f1 = 2 * tp / (predicted + positives)
    return float(f1.max()), int(len(groups))


def external_rules():
    """Both engines on the official splits under the grid rule (results.csv) and the ICE-ID rule."""
    results = pd.read_csv(OUT / "external/results.csv")
    rng = np.random.default_rng(np.random.SeedSequence(574))
    rows, boot = [], {}
    for dataset in sorted(results.dataset.unique()):
        valid = pd.read_csv(OUT / f"external/{dataset}/valid_scores.csv.gz")
        test = pd.read_csv(OUT / f"external/{dataset}/test_scores.csv.gz")
        vy, ty = valid.label.to_numpy(), test.label.to_numpy()
        decisions = {}
        for engine, column in ENGINES.items():
            r = results[results.dataset.eq(dataset) & results.model.eq(engine)].iloc[0]
            vs, ts = valid[column].to_numpy(float), test[column].to_numpy(float)
            grid = metrics(ty, ts, float(r.threshold))
            if abs(grid["f1"] - r.f1) > 5e-5:
                raise RuntimeError(f"{dataset}/{engine}: grid threshold does not reproduce results.csv")
            high = threshold_on_validation(vy, vs)
            low, n_tied = low_tie_threshold(vy, vs)
            if weighted_threshold(vy.astype(bool), vs, np.ones(len(vy), dtype=np.int64)) != high:
                raise RuntimeError("Weighted threshold selection does not reproduce threshold_on_validation")
            observed, observed_low = metrics(ty, ts, high), metrics(ty, ts, low)
            decisions[engine] = ts >= high
            row = {"dataset": dataset, "model": engine, "n_test": len(ty), "n_positive": int(ty.sum()),
                   "grid_threshold": float(r.threshold), "grid_valid_f1": metrics(vy, vs, float(r.threshold))["f1"],
                   "grid_f1": grid["f1"], "observed_threshold": high, "observed_low_threshold": low,
                   "observed_tied_thresholds": n_tied, "observed_valid_f1": metrics(vy, vs, high)["f1"],
                   "observed_low_f1": observed_low["f1"],
                   **{f"observed_{k}": observed[k] for k in ["precision", "recall", "f1", "tp", "fp", "fn", "tn"]}}
            rows.append(row)
        oracle, n_sets = set_oracle(ty, test.judgments.fillna("").to_numpy())
        for row in rows[-2:]:
            row.update({"judgment_set_oracle_f1": oracle, "test_judgment_sets": n_sets,
                        "engine_disagreements": int((decisions["ona"] != decisions["published_nars"]).sum())})
        # Paired pair-level bootstrap of ONA minus control, thresholds reselected on resampled validation pairs.
        deltas = np.empty(REPEATS)
        vy_b, ty_b = vy.astype(bool), ty.astype(bool)
        for i in range(REPEATS):
            vw = np.bincount(rng.integers(0, len(vy), len(vy)), minlength=len(vy))
            tw = np.bincount(rng.integers(0, len(ty), len(ty)), minlength=len(ty))
            if vw[vy_b].sum() == 0:
                deltas[i] = np.nan
                continue
            f = {}
            for engine, column in ENGINES.items():
                t = weighted_threshold(vy_b, valid[column].to_numpy(float), vw)
                f[engine] = weighted_f1(ty_b, test[column].to_numpy(float) >= t, tw)
            deltas[i] = f["ona"] - f["published_nars"]
        boot[dataset] = {"interval": np.nanquantile(deltas, [0.025, 0.975]).tolist(), "repeats": REPEATS,
                         "unusable_replicates": int(np.isnan(deltas).sum()),
                         "resampling": "pairs (validation and test), thresholds reselected; repeated records ignored"}
    return pd.DataFrame(rows), boot


# ------------------------------------------------------------------ decisions, movers, one record per wave
def iceid_oracle():
    """Best F1 of any judgment-set rule on each controlled test draw (in-sample, test labels)."""
    out = {}
    for seed in SEEDS:
        test = part(pairs(seed, "controlled"), "test")
        out[str(seed)] = set_oracle(test.label.to_numpy(), test.judgment_set.to_numpy())[0]
    return out


def decision_disagreements():
    """Pairs on which ONA and the control decide differently at their selected thresholds."""
    saved = {"controlled": pd.read_csv(TABLES / "metrics.csv").query("evaluation == 'controlled_crosswave_reference'"),
             "candidate": pd.read_csv(TABLES / "candidate_graph_metrics.csv")}
    locked = pd.read_csv(TABLES / "locked_metrics.csv")
    rows = []
    for seed in SEEDS:
        for population, stem, partition, key in [("controlled", "controlled", "test", "controlled"),
                                                 ("candidate", "candidate_graph", "test", "candidate"),
                                                 ("locked_controlled", "locked_controlled", "locked", "controlled"),
                                                 ("locked_candidate", "locked_candidate", "locked", "candidate")]:
            frame = part(pairs(seed, stem), partition)
            y = frame.label.to_numpy().astype(bool)
            d = {}
            for engine, column in ENGINES.items():
                t = float(saved[key].query("model == @engine and seed == @seed").threshold.iloc[0])
                d[engine] = frame[column].to_numpy(float) >= t
                ref = (saved[key] if not population.startswith("locked") else
                       locked.query("population == @key")).query("model == @engine and seed == @seed").iloc[0]
                if int((d[engine] & ~y).sum()) != int(ref.fp) or int((~d[engine] & y).sum()) != int(ref.fn):
                    raise RuntimeError(f"{seed}/{population}/{engine}: recomputed decisions differ from the saved table")
            rows.append({"seed": seed, "population": population, "pairs": len(y),
                         "disagree": int((d["ona"] != d["published_nars"]).sum()),
                         "ona_only_fp": int((d["ona"] & ~d["published_nars"] & ~y).sum()),
                         "control_only_fp": int((d["published_nars"] & ~d["ona"] & ~y).sum()),
                         "ona_only_tp": int((d["ona"] & ~d["published_nars"] & y).sum()),
                         "control_only_tp": int((d["published_nars"] & ~d["ona"] & y).sum())})
    return pd.DataFrame(rows)


class Geography:
    """Which true matches moved: commune (hreppur, the release's `county`) and county (sysla, its `district`)
    both differ, and neither difference is a recoding.

    The release's unit codes follow administrative history and are not guaranteed valid for a record's
    census year (ice-id README). A county code therefore also changes without a move when a county splits
    (Hunavatnssysla 1907, Bardastrandarsysla 1910) or when a whole commune's county code changes between
    censuses (Akureyri: Eyjafjardarsysla in 1901 and 1910, Sudur-Mulasysla in 1920). Within one census each
    commune carries one county code (checked for 1901-1920; earlier communes can carry two), so a commune is recoded between two censuses when its set of codes
    differs. A pair is a recoding when either commune is recoded between the pair's two censuses or any of
    its four codes lies outside the unit's validity years."""

    def __init__(self):
        from prepare_release import configured_path
        raw = configured_path("selected_release") / "raw_data"
        people = pd.read_csv(raw / "people.csv", usecols=["heimild", "county", "district"]).dropna()
        self.county_of = {key: frozenset(v.astype(int)) for key, v in people.groupby(["heimild", "county"]).district.unique().items()}
        if any(len(v) > 1 for (wave, _), v in self.county_of.items() if wave >= 1901):
            raise RuntimeError("A commune carries two county codes in one assessment census")
        self.units = {name: pd.read_csv(raw / f"{stem}.csv").set_index("id")[["begins", "ends"]]
                      for name, stem in [("district", "districts"), ("county", "counties")]}
        self.records = load_records()

    @staticmethod
    def code(record, field):
        value = record.get(field)
        return None if value is None or value == "" or (isinstance(value, float) and np.isnan(value)) else int(value)

    def valid(self, field, code, year):
        if code is None:
            return True
        if code not in self.units[field].index:
            return False
        begins, ends = self.units[field].loc[code]
        return (pd.isna(begins) or begins <= year) and (pd.isna(ends) or year <= ends)

    def classify(self, id1, id2):
        """'mover', 'recoded' (county code differs without a move) or 'other'; also whether the county code differs."""
        a, b = self.records[int(id1)], self.records[int(id2)]
        wa, wb = int(a["heimild"]), int(b["heimild"])
        da, db, ca, cb = (self.code(r, f) for r, f in [(a, "district"), (b, "district"), (a, "county"), (b, "county")])
        county_differs = da is not None and db is not None and da != db
        commune_differs = ca is not None and cb is not None and ca != cb
        if not county_differs:
            return "other", False
        recoded_commune = any(self.county_of[(wa, c)] != self.county_of[(wb, c)] for c in (ca, cb)
                              if c is not None and (wa, c) in self.county_of and (wb, c) in self.county_of)
        invalid = not all(self.valid(f, code, year) for f, code, year in
                          [("district", da, wa), ("district", db, wb), ("county", ca, wa), ("county", cb, wb)])
        return ("mover" if commune_differs and not recoded_commune and not invalid else "recoded"), True

    def frame(self, cases):
        out = [self.classify(i, j) for i, j in zip(cases.id1, cases.id2)]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def movers_and_waves():
    """Recall for true matches that moved (commune and county both changed, recodings excluded; Geography) versus
    all other true matches, and partitions that put two records of one census wave into one cluster (candidate
    graphs, test role). Also the movers' share of training matches under both training protocols."""
    sets = pd.read_csv(ICE / "judgment_sets.csv.gz").set_index("judgment_set").judgments.str.split().map(frozenset)
    geography = Geography()
    saved = pd.read_csv(CAND / "graph_metrics.csv")
    ona_saved = pd.read_csv(TABLES / "candidate_graph_metrics.csv")
    lr_saved = pd.read_csv(TABLES / "judgment_lr_metrics.csv").query("population == 'candidate_graph' and training == 'candidate_matched'")
    movers, waves = [], []
    for seed in SEEDS:
        graph = part(pairs(seed, "candidate_graph"), "test")
        predictions = pd.read_csv(CAND / f"seed_{seed}/graph_predictions.csv.gz", float_precision="round_trip")
        if not predictions[["id1", "id2", "label"]].equals(graph[["id1", "id2", "label"]]):
            raise RuntimeError("Candidate predictions and ONA graph pairs differ")
        node_frame = pd.read_csv(PUB / f"seed_{seed}/test_graph_nodes.csv").set_index("id", drop=False)
        y = graph.label.to_numpy().astype(bool)
        kind, county_differs = geography.frame(graph)
        judged = graph.judgment_set.map(lambda s: "different_district" in sets[s]).to_numpy()
        if not np.array_equal(county_differs[y], judged[y]):
            raise RuntimeError(f"seed {seed}: county codes and the different_district judgment disagree on true matches")
        moved, stayed, recoded = kind == "mover", kind != "mover", kind == "recoded"
        if not graph.judgment_set[y & moved].map(lambda s: {"different_county", "different_district"} <= sets[s]).all():
            raise RuntimeError(f"seed {seed}: a mover's judgments do not show a different commune and county")
        nested = graph.judgment_set.map(lambda s: {"different_farm", "different_parish", "different_county",
                                                   "different_district"} <= sets[s]).to_numpy()
        scored = {}
        for engine, column in ENGINES.items():
            t = float(ona_saved.query("model == @engine and seed == @seed").threshold.iloc[0])
            scored[engine] = (graph[column].to_numpy(float), t, None)
        for name in CLASSICAL:
            t = json.loads((CAND / f"seed_{seed}/{name}_decisions.json").read_text())["threshold"]
            clusters = pd.read_csv(CAND / f"seed_{seed}/{name}_clusters.csv.gz")
            scored[name] = (predictions[name].to_numpy(float), t, clusters)
        values, t = judgment_lr(seed, "candidate_test", graph)
        scored[JUDGMENT_LR] = (values, t, pd.read_csv(JUDGMENT_LR_DIR / f"seed_{seed}/candidate_test_clusters.csv.gz"))
        for name, (values, t, clusters) in scored.items():
            decide = values >= t
            if name == JUDGMENT_LR:
                ref = lr_saved.query("seed == @seed")
            else:
                ref = (ona_saved if name in ENGINES else saved).query("model == @name and seed == @seed")
            if int((decide & ~y).sum()) != int(ref.fp.iloc[0]) or int((~decide & y).sum()) != int(ref.fn.iloc[0]):
                raise RuntimeError(f"{seed}/{name}: recomputed graph decisions differ from the saved metrics")
            fn = ~decide & y
            top = graph.judgment_set[fn].value_counts()
            movers.append({"seed": seed, "model": name, "positives": int(y.sum()), "movers": int((y & moved).sum()),
                           "stayers": int((y & stayed).sum()), "unknown": int((y & ~moved & ~stayed).sum()),
                           "county_code_changed": int((y & county_differs).sum()), "recoded": int((y & recoded).sum()),
                           "movers_all_four_differ": int((y & moved & nested).sum()),
                           "recall_movers": float(decide[y & moved].mean()), "recall_stayers": float(decide[y & stayed].mean()),
                           "recall_recoded": float(decide[y & recoded].mean()),
                           "fn": int(fn.sum()), "fn_movers": int((fn & moved).sum()),
                           "fn_top_set": top.index[0] if len(top) else "", "fn_top_set_count": int(top.iloc[0]) if len(top) else 0,
                           "fn_top_set_score": float(values[fn & graph.judgment_set.eq(top.index[0]).to_numpy()].max()) if len(top) else np.nan,
                           "threshold": float(t)})
            if clusters is None:
                _, clusters = full_node_clusters(node_frame, graph, values, t)
            merged = clusters.merge(node_frame[["id", "heimild"]].reset_index(drop=True), on="id", validate="one_to_one")
            size = merged.groupby("predicted_cluster").id.transform("size")
            if int(size.max()) != int(ref.largest_component.iloc[0]):
                raise RuntimeError(f"{seed}/{name}: saved clusters differ from the reported largest component")
            repeated = merged.groupby("predicted_cluster").heimild.transform(lambda h: h.duplicated(keep=False).any())
            waves.append({"seed": seed, "model": name, "records": len(merged), "largest_cluster": int(size.max()),
                          "clusters_with_repeated_wave": int(merged[repeated].predicted_cluster.nunique()),
                          "records_in_such_clusters": int(repeated.sum()),
                          "reference_max_per_person_wave": int(node_frame.groupby(["person", "heimild"]).size().max())})
    shares = []
    for seed in SEEDS:
        controlled = part(pairs(seed, "controlled"), "train")
        matched = pd.read_csv(CAND / f"seed_{seed}/train_pairs.csv")
        for training, frame in [("controlled", controlled), ("candidate_matched", matched)]:
            matches = frame[frame.label.eq(1)]
            kind, county_differs = geography.frame(matches)
            shares.append({"seed": seed, "training": training, "matches": len(matches), "movers": int((kind == "mover").sum()),
                           "recoded": int((kind == "recoded").sum()), "county_code_changed": int(county_differs.sum())})
    return pd.DataFrame(movers), pd.DataFrame(waves), pd.DataFrame(shares)


def birthyear_veto():
    """ONA with a veto on the one judgment the label screens make a perfect non-match signal.

    The screens withhold every identity whose records span more than five birth years, so no retained match
    carries `different_birthyear`. The counterfactual keeps ONA's scores and selected thresholds and rejects
    every pair with that judgment; it is compared with ONA and with logistic regression over ONA's judgments."""
    sets = pd.read_csv(ICE / "judgment_sets.csv.gz").set_index("judgment_set").judgments.str.split().map(frozenset)
    knowledge = pd.read_csv(TABLES / "knowledge.csv")
    saved = {"controlled": pd.read_csv(TABLES / "metrics.csv").query("evaluation == 'controlled_crosswave_reference'"),
             "candidate_graph": pd.read_csv(TABLES / "candidate_graph_metrics.csv")}
    rows = []
    for seed in SEEDS:
        for population, stem, kb, lr_stem in [("controlled", "controlled", "pub", "controlled_test"),
                                              ("candidate_graph", "candidate_graph", "cand", "candidate_test")]:
            test = part(pairs(seed, stem), "test")
            y = test.label.to_numpy().astype(bool)
            vetoed = test.judgment_set.map(lambda s: "different_birthyear" in sets[s]).to_numpy()
            ref = saved[population].query("model == 'ona' and seed == @seed").iloc[0]
            ona = test.ona_score.to_numpy(float) >= float(ref.threshold)
            if int((ona & ~y).sum()) != int(ref.fp) or int((~ona & y).sum()) != int(ref.fn):
                raise RuntimeError(f"{seed}/{population}: recomputed ONA decisions differ from the saved metrics")
            values, t = judgment_lr(seed, lr_stem, test)
            lr = values >= t
            k = knowledge.query("seed == @seed and knowledge_base == @kb and judgment == 'different_birthyear'").iloc[0]
            f1 = lambda d: 2 * (d & y).sum() / (2 * (d & y).sum() + (d & ~y).sum() + (~d & y).sum())
            rows.append({"seed": seed, "population": population, "train_pairs_with_judgment": int(k.w),
                         "train_matches_with_judgment": int(k.w_plus), "test_positives_with_judgment": int((y & vetoed).sum()),
                         "ona_f1": float(f1(ona)), "ona_fp": int((ona & ~y).sum()), "ona_fp_with_judgment": int((ona & ~y & vetoed).sum()),
                         "veto_f1": float(f1(ona & ~vetoed)), "veto_fp": int((ona & ~vetoed & ~y).sum()), "veto_fn": int((~(ona & ~vetoed) & y).sum()),
                         "lr_judgments_f1": float(f1(lr)), "lr_judgments_fp": int((lr & ~y).sum()),
                         "lr_judgments_fp_with_judgment": int((lr & ~y & vetoed).sum())})
    return pd.DataFrame(rows)


SETCOUNT = BASE / "experiments/ona/setcount"


def setcount_movers(geography=None):
    """Candidate-graph recall of judgment-set ONA (run_ona_setcount.py, revision design) for movers under the
    Geography rule, so it is comparable with graph_movers.csv. Each pair's score is the expectation of ONA's
    answer for its judgment set under the candidate-matched knowledge; the selected threshold and the
    resulting FP/FN must equal the set-count metrics table."""
    geography = geography or Geography()
    answers = pd.read_csv(SETCOUNT / "tables/episodes_revision.csv.gz").query("kb == 'cand'")
    saved = pd.read_csv(SETCOUNT / "tables/metrics.csv", float_precision="round_trip").query(
        "population == 'candidate_graph' and model == 'ona_set_revision'")
    rows = []
    for seed in SEEDS:
        graph = part(pairs(seed, "candidate_graph"), "test")
        score = answers[answers.seed.eq(seed)].set_index("judgment_set").expectation
        values = graph.judgment_set.map(score).to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"seed {seed}: a candidate-graph pair has no judgment-set ONA answer")
        ref = saved.query("seed == @seed").iloc[0]
        y, decide = graph.label.to_numpy().astype(bool), values >= float(ref.threshold)
        if int((decide & ~y).sum()) != int(ref.fp) or int((~decide & y).sum()) != int(ref.fn):
            raise RuntimeError(f"seed {seed}: recomputed judgment-set ONA decisions differ from the set-count metrics")
        kind, _ = geography.frame(graph)
        moved = kind == "mover"
        rows.append({"seed": seed, "model": "ona_set_revision", "positives": int(y.sum()), "movers": int((y & moved).sum()),
                     "recall_movers": float(decide[y & moved].mean()), "recall_stayers": float(decide[y & ~moved].mean()),
                     "fn": int((~decide & y).sum()), "fn_movers": int((~decide & y & moved).sum()), "threshold": float(ref.threshold)})
    return pd.DataFrame(rows)


def judgment_diagnostics():
    """The mover, recoding and birth-year-veto outputs only (no bootstrap); main() writes them too."""
    movers, waves, shares = movers_and_waves()
    movers.to_csv(TABLES / "graph_movers.csv", index=False)
    waves.to_csv(TABLES / "graph_wave_repeats.csv", index=False)
    shares.to_csv(TABLES / "mover_training_shares.csv", index=False)
    birthyear_veto().to_csv(TABLES / "birthyear_veto.csv", index=False)
    setcount_movers().to_csv(TABLES / "setcount_graph_movers.csv", index=False)


def main():
    from resource_limits import check_memory
    check_memory(2)
    if sys.argv[1:] == ["--judgment-diagnostics"]:
        judgment_diagnostics()
        return
    if sys.argv[1:] == ["--setcount-movers"]:
        setcount_movers().to_csv(TABLES / "setcount_graph_movers.csv", index=False)
        return
    persons = load_persons()
    main_boot, reproduced = bootstrap_main(persons)
    locked_boot = bootstrap_locked(persons)
    ties = tie_sensitivity()
    ties.to_csv(TABLES / "ice_tie_sensitivity.csv", index=False)
    reliability().to_csv(TABLES / "reliability_bins.csv", index=False)
    label_audit_engines().to_csv(TABLES / "label_audit_engines.csv", index=False)
    rules, external_boot = external_rules()
    rules.to_csv(TABLES / "external_threshold_rules.csv", index=False)
    decision_disagreements().to_csv(TABLES / "engine_disagreements.csv", index=False)
    judgment_diagnostics()
    result = {"method": __doc__.strip().splitlines()[0],
              "seed131_reproduced_interval": reproduced,
              "controlled_main": main_boot, "controlled_locked": locked_boot,
              "reselected_threshold_bootstrap": bootstrap_reselected(persons),
              "external_paired_bootstrap": external_boot, "iceid_judgment_set_oracle": iceid_oracle(),
              "candidate_graph_bootstrap": "not computed: the evaluated candidate pairs form one connected identity component",
              "score_facts": score_facts()}
    dump(TABLES / "engine_comparison.json", result)
    v = main_boot["versus_control"]["ona"]
    print(json.dumps({"seed131_reproduced": reproduced, "ona_minus_control_3seed": v["mean_delta"],
                      "interval": v["mean_delta_interval"],
                      "locked": locked_boot["versus_control"]["ona"]["mean_delta_interval"]}, indent=1))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main()
