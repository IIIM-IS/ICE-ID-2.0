"""ICE-ID tables for ONA, with the additive NARS/LLR control recomputed by the same code.

Reads only the per-pair score files of run_ona_full.py iceid (experiments/ona/full/iceid/pairs)
plus the saved node files, and calls the existing protocol functions
(run_experiments.metrics/threshold_on_validation, publication_run.fit_calibrators/
calibrated_scores, publication_evaluation.proper_scores/full_node_clusters/
tie_averaged_precision_at_positive_count/component_bootstrap). Every LLR row is compared with
the published row it mirrors (llr_reproduction.json); a mismatch stops the run. ONA rows are
written with model "ona" (ablations "ona_ablate_<category>"), LLR rows with the original names.

Thresholds: selection partition only (controlled and graph); calibration maps: calibration
partition only; training-F1 policy: training pairs only; locked assessment: no refit, the
development thresholds and maps of the same engine. A cell whose population contains an
unanswered ONA pair is not computed (reported), never filled.
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
from publication_run import fit_calibrators, calibrated_scores  # noqa: E402
from publication_evaluation import (proper_scores, full_node_clusters,  # noqa: E402
                                    tie_averaged_precision_at_positive_count, component_bootstrap)
from run_ona_full import ICE, SEEDS, PUB, CAND, LOCKED, CATEGORIES, dump, load_persons  # noqa: E402

TABLES = ICE / "tables"
ENGINES = {"ona": "ona_score", "published_nars": "llr_score"}
NEURAL = BASE / "experiments/neural-quarantined-extended"
AAAI_DATA = BASE.parent / "new_paper_subs/AAAI Magazine (model paper)/figure_data"
SKIP = {"fit_seconds", "elapsed_seconds", "score_seconds"}   # timing, not results
REPRO = []


class Unanswered(Exception):
    pass


def pairs(seed, stem):
    return pd.read_csv(ICE / f"pairs/seed_{seed}/{stem}.csv.gz", float_precision="round_trip")


def part(frame, partition):
    return frame[frame.partition.eq(partition)].reset_index(drop=True)


def scores(frame, engine):
    values = frame[ENGINES[engine]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise Unanswered(f"{int((~np.isfinite(values)).sum())} unanswered pairs")
    return values


INTEGER_FIELDS = {"tp", "fp", "tn", "fn", "n_pairs", "n_positive", "provisional_different_linked",
                  "provisional_different_count", "excluded_pairs", "n_nodes", "n_predicted_clusters", "n_isolates",
                  "largest_component"}


def compare(kind, expected: dict, actual: dict, keys=None, tolerance=1e-9):
    """Record and enforce that a recomputed LLR row equals the published one.

    Integer fields must be identical. Float fields within ``tolerance``: 1e-9 in general;
    rows that refit a sigmoid map (LBFGS logistic regression, C=1e6) get 1e-6 because the
    optimiser's last digits depend on thread count. The actual differences are recorded.
    """
    keys = keys or [k for k in expected if k in actual and k not in SKIP]
    diffs = {}
    for k in keys:
        e, a = expected[k], actual[k]
        if isinstance(e, str) or isinstance(a, str):
            same = str(e) == str(a)
            diffs[k] = 0.0 if same else float("inf")
        elif e is None or a is None or (isinstance(e, float) and np.isnan(e)):
            same = (e is None or np.isnan(e)) and (a is None or (isinstance(a, float) and np.isnan(a)))
            diffs[k] = 0.0 if same else float("inf")
        else:
            diffs[k] = abs(float(e) - float(a))
    worst = max(diffs.values()) if diffs else 0.0
    REPRO.append({"row": kind, "fields": len(diffs), "max_abs_difference": worst, "tolerance": tolerance,
                  "integer_fields_identical": all(diffs[k] == 0 for k in diffs if k in INTEGER_FIELDS)})
    bad = {k: v for k, v in diffs.items() if (v != 0 if k in INTEGER_FIELDS else v > tolerance)}
    if bad:
        raise RuntimeError(f"LLR recomputation differs from the published row {kind}: {bad}")


def saved_row(frame, **where):
    mask = np.ones(len(frame), dtype=bool)
    for k, v in where.items():
        mask &= frame[k].eq(v).to_numpy()
    rows = frame[mask]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one published row for {where}, found {len(rows)}")
    return {k: (None if pd.isna(v) else v) for k, v in rows.iloc[0].to_dict().items()}


def nodes(path):
    return pd.read_csv(path).set_index("id", drop=False)


# ------------------------------------------------------------------ development (publication run)
def publication(seed, saved, out, state, missing):
    controlled = pairs(seed, "controlled")
    audit = part(controlled, "audit")
    graph = pairs(seed, "publication_graph")
    test_nodes = nodes(PUB / f"seed_{seed}/test_graph_nodes.csv")
    fits = json.loads((ICE / "fits.json").read_text())[str(seed)]
    for engine in ENGINES:
        try:
            sel, cal, test, train = (scores(part(controlled, p), engine) for p in ["selection", "calibration", "test", "train"])
            aud = scores(audit, engine)
        except Unanswered as error:
            missing.append({"seed": seed, "engine": engine, "table": "controlled", "reason": str(error)})
            continue
        labels = {p: part(controlled, p).label.to_numpy() for p in ["selection", "calibration", "test", "train"]}
        threshold = threshold_on_validation(labels["selection"], sel)
        state[(seed, engine, "controlled_threshold")] = threshold
        for condition, y in [("controlled_crosswave_reference", labels["test"]),
                             ("audit_provisional_reference", audit.label.to_numpy()),
                             ("audit_inherited_reference", np.ones(len(audit), dtype=int))]:
            s = test if condition.startswith("controlled") else aud
            row = {"seed": seed, "training": "corrected_training_labels", "model": engine, "evaluation": condition,
                   **metrics(y, s, threshold)}
            if condition.startswith("audit"):
                row["provisional_different_linked"] = int(np.sum((aud >= threshold) & audit.label.eq(0)))
                row["provisional_different_count"] = int(audit.label.eq(0).sum())
            out["metrics"].append(row)
            if engine == "published_nars":
                compare(f"metrics/{seed}/{condition}", saved_row(saved["metrics"], seed=seed, model=engine,
                        evaluation=condition, training="corrected_training_labels"), row)
        # Audit source-timing populations (make_paper_artifacts.audit_source_timing_sensitivity masks).
        uncertain = ((audit.id1.eq(64385) & audit.id2.eq(104931)) | (audit.id1.eq(104931) & audit.id2.eq(64385)))
        if int(uncertain.sum()) != 1 or len(audit) != 136 or int(audit.label.sum()) != 19:
            raise RuntimeError("Audit population differs from the reported one")
        for population, mask in [("full_provisional", np.ones(len(audit), dtype=bool)),
                                 ("exclude_nominal1816", audit.wave.ne(1816).to_numpy()),
                                 ("exclude_source_uncertain_jon", ~uncertain.to_numpy())]:
            y, s = audit.label.to_numpy()[mask], aud[mask]
            row = {"seed": seed, "model": engine, "population": population, **metrics(y, s, threshold),
                   "provisional_different_count": int((y == 0).sum()),
                   "provisional_different_linked": int(np.sum((y == 0) & (s >= threshold))),
                   "excluded_pairs": int(len(audit) - mask.sum())}
            out["audit_source_timing"].append(row)
            if engine == "published_nars":
                compare(f"audit_source_timing/{seed}/{population}",
                        saved_row(saved["audit_source_timing"], seed=seed, model=engine, population=population), row)
        # Threshold policies (publication_replay): fixed 0.5, training F1, selection F1.
        for policy, t in [("fixed_0.5", .5), ("training_f1", threshold_on_validation(labels["train"], train)),
                          ("selection_f1", threshold)]:
            row = {"seed": seed, "training": "corrected_training_labels", "policy": policy, "model": engine,
                   **metrics(labels["test"], test, t)}
            out["threshold_sensitivity"].append(row)
            if engine == "published_nars":
                compare(f"threshold_sensitivity/{seed}/{policy}",
                        saved_row(saved["threshold_sensitivity"], seed=seed, policy=policy), row)
        maps = fit_calibrators(labels["calibration"], cal)
        state[(seed, engine, "controlled_maps")] = maps
        for method, calibrator in {"raw": None, **maps}.items():
            t_test = test if calibrator is None else calibrated_scores(test, calibrator)
            t_sel = sel if calibrator is None else calibrated_scores(sel, calibrator)
            operating = threshold_on_validation(labels["selection"], t_sel)
            state[(seed, engine, "controlled_policy", method)] = operating
            row = {"seed": seed, "training": "corrected_training_labels", "model": engine,
                   "population": "controlled_2to1", "calibration": method,
                   **metrics(labels["test"], t_test, operating), **proper_scores(labels["test"], t_test)}
            out["calibration_metrics"].append(row)
            if engine == "published_nars":
                compare(f"calibration/{seed}/controlled/{method}", saved_row(saved["calibration"], seed=seed, model=engine,
                        population="controlled_2to1", calibration=method, training="corrected_training_labels"), row, tolerance=1e-6 if method == "sigmoid" else 1e-9)
        # Publication-trained candidate graph (graph_metrics.csv) and its calibration rows.
        try:
            g = {p: scores(part(graph, p), engine) for p in ["selection", "calibration", "test"]}
        except Unanswered as error:
            missing.append({"seed": seed, "engine": engine, "table": "publication_graph", "reason": str(error)})
            continue
        gl = {p: part(graph, p).label.to_numpy() for p in g}
        gt = threshold_on_validation(gl["selection"], g["selection"])
        gmaps = fit_calibrators(gl["calibration"], g["calibration"])
        for method, calibrator in {"raw": None, **gmaps}.items():
            t_test = g["test"] if calibrator is None else calibrated_scores(g["test"], calibrator)
            t_sel = g["selection"] if calibrator is None else calibrated_scores(g["selection"], calibrator)
            operating = threshold_on_validation(gl["selection"], t_sel)
            row = {"seed": seed, "training": "corrected_training_labels", "model": engine,
                   "population": "candidate_graph", "calibration": method,
                   **metrics(gl["test"], t_test, operating), **proper_scores(gl["test"], t_test)}
            out["calibration_metrics"].append(row)
            if engine == "published_nars":
                compare(f"calibration/{seed}/graph/{method}", saved_row(saved["calibration"], seed=seed, model=engine,
                        population="candidate_graph", calibration=method, training="corrected_training_labels"), row, tolerance=1e-6 if method == "sigmoid" else 1e-9)
        test_graph = part(graph, "test")
        published = saved_row(saved["graph"], seed=seed, model="published_nars", training="corrected_training_labels")
        if published["n_candidates"] != len(test_graph) or published["candidate_positives"] != int(test_graph.label.sum()):
            raise RuntimeError("Candidate graph pair counts differ from the LLR run")
        cluster, _ = full_node_clusters(test_nodes, test_graph, g["test"], gt)
        blocking = {k: published[k] for k in ["n_records", "n_persons", "n_adjacent_reference_positives", "method",
                                              "max_cap", "selection", "cap", "n_candidates", "candidate_positives",
                                              "candidates_per_record", "blocking_recall", "positive_rate"]}
        row = {"seed": seed, "training": "corrected_training_labels", "model": engine, **blocking,
               **metrics(gl["test"], g["test"], gt), **cluster,
               "p_at_reference_positive_count": tie_averaged_precision_at_positive_count(gl["test"], g["test"]),
               "fit_seconds": fits["pub"]["fit_seconds"]}
        out["graph_metrics"].append(row)
        if engine == "published_nars":
            compare(f"graph/{seed}", published, row)


def ablations(seed, saved, out, missing):
    for c in CATEGORIES:
        frame = pairs(seed, f"ablate_{c}")
        for engine in ENGINES:
            name = ("ona_" if engine == "ona" else "") + f"ablate_{c}"
            try:
                sel, test = scores(part(frame, "selection"), engine), scores(part(frame, "test"), engine)
            except Unanswered as error:
                missing.append({"seed": seed, "engine": engine, "table": name, "reason": str(error)})
                continue
            t = threshold_on_validation(part(frame, "selection").label.to_numpy(), sel)
            row = {"seed": seed, "model": name, **metrics(part(frame, "test").label.to_numpy(), test, t)}
            out["ablations"].append(row)
            if engine == "published_nars":
                compare(f"ablations/{seed}/{c}", saved_row(saved["ablations"], seed=seed, model=name), row)


def candidate(seed, saved, out, state, missing):
    graph = pairs(seed, "candidate_graph")
    raw_nodes = nodes(PUB / f"seed_{seed}/test_graph_nodes.csv")
    for engine in ENGINES:
        try:
            g = {p: scores(part(graph, p), engine) for p in ["selection", "calibration", "test"]}
        except Unanswered as error:
            missing.append({"seed": seed, "engine": engine, "table": "candidate_graph", "reason": str(error)})
            continue
        gl = {p: part(graph, p).label.to_numpy() for p in g}
        t = threshold_on_validation(gl["selection"], g["selection"])
        state[(seed, engine, "candidate_threshold")] = t
        cluster, _ = full_node_clusters(raw_nodes, part(graph, "test"), g["test"], t)
        row = {"seed": seed, "model": engine, "training": "candidate_matched_training",
               **metrics(gl["test"], g["test"], t), **cluster}
        out["candidate_graph_metrics"].append(row)
        if engine == "published_nars":
            compare(f"candidate_graph/{seed}", saved_row(saved["candidate_graph"], seed=seed, model=engine), row)
        out["candidate_ranking"].append({"seed": seed, "model": engine, "k": int(gl["test"].sum()),
                                         "p_at_k": tie_averaged_precision_at_positive_count(gl["test"], g["test"])})
        maps = fit_calibrators(gl["calibration"], g["calibration"])
        state[(seed, engine, "candidate_maps")] = maps
        for method, calibrator in {"raw": None, **maps}.items():
            ts = g["test"] if calibrator is None else calibrated_scores(g["test"], calibrator)
            vs = g["selection"] if calibrator is None else calibrated_scores(g["selection"], calibrator)
            operating = threshold_on_validation(gl["selection"], vs)
            state[(seed, engine, "candidate_policy", method)] = operating
            row = {"seed": seed, "model": engine, "calibration": method,
                   **metrics(gl["test"], ts, operating), **proper_scores(gl["test"], ts)}
            out["candidate_calibration_metrics"].append(row)
            if engine == "published_nars":
                compare(f"candidate_calibration/{seed}/{method}",
                        saved_row(saved["candidate_calibration"], seed=seed, model=engine, calibration=method), row, tolerance=1e-6 if method == "sigmoid" else 1e-9)


def locked(seed, saved, out, state, missing, bootstrap):
    locked_nodes = nodes(LOCKED / "candidate_nodes.csv")
    for population, stem, key in [("controlled", "locked_controlled", "controlled"),
                                  ("candidate", "locked_candidate", "candidate")]:
        frame = pairs(seed, stem)
        for engine in ENGINES:
            if (seed, engine, f"{key}_threshold") not in state:
                missing.append({"seed": seed, "engine": engine, "table": f"locked_{population}",
                                "reason": "development threshold not computed"})
                continue
            try:
                values = scores(frame, engine)
            except Unanswered as error:
                missing.append({"seed": seed, "engine": engine, "table": f"locked_{population}", "reason": str(error)})
                continue
            y = frame.label.to_numpy()
            threshold = state[(seed, engine, f"{key}_threshold")]
            row = {"seed": seed, "population": population, "model": engine, **metrics(y, values, threshold),
                   **proper_scores(y, values)}
            if population == "candidate":
                partition, _ = full_node_clusters(locked_nodes, frame, values, threshold)
                row.update(partition)
                row["p_at_k"] = tie_averaged_precision_at_positive_count(y, values)
            out["locked_metrics"].append(row)
            if engine == "published_nars":
                compare(f"locked/{seed}/{population}", saved_row(saved["locked"], seed=seed, population=population,
                                                                 model=engine), row)
            for method, calibrator in state[(seed, engine, f"{key}_maps")].items():
                mapped = calibrated_scores(values, calibrator)
                row = {"seed": seed, "population": population, "model": engine, "calibration": method,
                       **metrics(y, mapped, state[(seed, engine, f"{key}_policy", method)]), **proper_scores(y, mapped)}
                out["locked_calibration_metrics"].append(row)
                if engine == "published_nars":
                    compare(f"locked_calibration/{seed}/{population}/{method}",
                            saved_row(saved["locked_calibration"], seed=seed, population=population, model=engine,
                                      calibration=method), row, tolerance=1e-6 if method == "sigmoid" else 1e-9)
            if seed == SEEDS[0] and population == "controlled":
                bootstrap[engine] = (frame, values, threshold)


def bootstraps(state, out_dir, missing):
    """Seed-131 conditional component bootstraps (publication test and locked controlled).

    component_bootstrap draws one weight vector per resample regardless of how many models it is
    given (rng seed 571), so adding ONA leaves every other interval unchanged; the saved intervals
    are reproduced. A second call names ONA as the reference, giving every model's paired delta
    versus ONA on the same resamples.
    """
    persons = load_persons()
    seed = SEEDS[0]
    vd = PUB / f"seed_{seed}/corrected_training_labels"
    test = pd.read_csv(vd / "test_predictions.csv", float_precision="round_trip")
    classical = ["published_nars", "fellegi_sunter", "rules", "logistic_regression", "gradient_boosting", "mlp"]
    results = {}
    mine = part(pairs(seed, "controlled"), "test")
    if not mine[["id1", "id2", "label"]].equals(test[["id1", "id2", "label"]]):
        raise RuntimeError("Bootstrap cases differ from the saved test predictions")
    for name, cases, predictions, thresholds, saved_path in [
            ("publication_test", test, {n: test[n].to_numpy() for n in classical},
             {n: json.loads((vd / f"{n}_decisions.json").read_text())["controlled_threshold"] for n in classical},
             PUB / "uncertainty.json"),
            ("locked_controlled", None, {}, {}, LOCKED / "uncertainty.json")]:
        if name == "locked_controlled":
            cases = pd.read_csv(LOCKED / "controlled_pairs.csv")
            for n in classical + ["ditto", "hiergat"]:
                saved = pd.read_csv(LOCKED / f"seed_{seed}/controlled_{n}_predictions.csv.gz", float_precision="round_trip")
                if not saved[["id1", "id2", "label"]].equals(cases[["id1", "id2", "label"]]):
                    raise RuntimeError("Locked predictions differ in cases")
                predictions[n] = saved.score.to_numpy()
                thresholds[n] = (json.loads((NEURAL / f"seed_{seed}/{n}/decisions.json").read_text())["threshold"]
                                 if n in ["ditto", "hiergat"] else
                                 json.loads((vd / f"{n}_decisions.json").read_text())["controlled_threshold"])
            ona_values = state.get((seed, "locked_ona_values"))
        else:
            ona_values = mine.ona_score.to_numpy()
        if ona_values is None or not np.isfinite(ona_values).all() or (seed, "ona", "controlled_threshold") not in state:
            missing.append({"seed": seed, "engine": "ona", "table": f"bootstrap_{name}", "reason": "ONA scores incomplete"})
            continue
        ona_t = state[(seed, "ona", "controlled_threshold")]
        people = persons
        standard = component_bootstrap(cases, people, {**predictions, "ona": ona_values}, {**thresholds, "ona": ona_t})
        saved = json.loads(saved_path.read_text())
        for model, values in saved["models"].items():
            got = standard["models"][model]
            if not (np.allclose(got["f1_interval"], values["f1_interval"], rtol=0, atol=1e-12) and
                    np.allclose(got["paired_delta_vs_llr_interval"], values["paired_delta_vs_llr_interval"], rtol=0, atol=1e-12)):
                raise RuntimeError(f"Bootstrap of {model} ({name}) differs from the saved intervals")
        REPRO.append({"row": f"bootstrap/{name}", "fields": len(saved["models"]), "max_abs_difference": 0.0})
        renamed = {("published_nars" if n == "ona" else ("llr" if n == "published_nars" else n)): v
                   for n, v in {**predictions, "ona": ona_values}.items()}
        rethresh = {("published_nars" if n == "ona" else ("llr" if n == "published_nars" else n)): v
                    for n, v in {**thresholds, "ona": ona_t}.items()}
        versus = component_bootstrap(cases, people, renamed, rethresh)
        versus_ona = {("ona" if n == "published_nars" else ("published_nars" if n == "llr" else n)):
                      {"f1_interval": v["f1_interval"], "paired_delta_vs_ona_interval": v["paired_delta_vs_llr_interval"]}
                      for n, v in versus["models"].items()}
        results[name] = {"standard_reference_llr": standard, "reference_ona": {
            "status": versus["status"], "n_components": versus["n_components"], "n_resamples": versus["n_resamples"],
            "models": versus_ona, "note": "same 500 resamples (rng 571); delta = model F1 minus ONA F1"}}
    dump(out_dir / "uncertainty.json", results)


def run():
    from resource_limits import check_memory
    check_memory(3)
    saved = {"metrics": pd.read_csv(PUB / "metrics.csv", float_precision="round_trip"),
             "graph": pd.read_csv(PUB / "graph_metrics.csv", float_precision="round_trip"),
             "calibration": pd.read_csv(PUB / "calibration_metrics.csv", float_precision="round_trip"),
             "ablations": pd.read_csv(PUB / "ablations.csv", float_precision="round_trip"),
             "threshold_sensitivity": pd.read_csv(PUB / "threshold_sensitivity.csv", float_precision="round_trip"),
             "candidate_graph": pd.read_csv(CAND / "graph_metrics.csv", float_precision="round_trip"),
             "candidate_calibration": pd.read_csv(CAND / "calibration_metrics.csv", float_precision="round_trip"),
             "locked": pd.read_csv(LOCKED / "metrics.csv", float_precision="round_trip"),
             "locked_calibration": pd.read_csv(LOCKED / "calibration_metrics.csv", float_precision="round_trip"),
             "audit_source_timing": pd.read_csv(AAAI_DATA / "audit_source_timing.csv", float_precision="round_trip")}
    names = ["metrics", "audit_source_timing", "threshold_sensitivity", "calibration_metrics", "graph_metrics",
             "ablations", "candidate_graph_metrics", "candidate_ranking", "candidate_calibration_metrics",
             "locked_metrics", "locked_calibration_metrics"]
    out, state, missing, boot = {n: [] for n in names}, {}, [], {}
    for seed in SEEDS:
        publication(seed, saved, out, state, missing)
        ablations(seed, saved, out, missing)
        candidate(seed, saved, out, state, missing)
        locked(seed, saved, out, state, missing, boot)
        print(f"seed {seed} tables done", flush=True)
    if "ona" in boot:
        state[(SEEDS[0], "locked_ona_values")] = boot["ona"][1]
    TABLES.mkdir(parents=True, exist_ok=True)
    bootstraps(state, TABLES, missing)
    for name in names:
        pd.DataFrame(out[name]).to_csv(TABLES / f"{name}.csv", index=False)
    knowledge = json.loads((ICE / "knowledge.json").read_text())
    rows = [{"seed": int(seed), "knowledge_base": kb, **{k: v for k, v in r.items() if k != "narsese"}, "narsese": r["narsese"]}
            for seed, kbs in knowledge.items() for kb, rs in kbs.items() for r in rs]
    pd.DataFrame(rows).to_csv(TABLES / "knowledge.csv", index=False)
    dump(TABLES / "llr_reproduction.json", {"rows_compared": len(REPRO), "max_abs_difference": max(r["max_abs_difference"] for r in REPRO),
                                           "rows": REPRO, "missing_cells": missing})
    print(json.dumps({"llr_rows_reproduced": len(REPRO), "max_abs_difference": max(r["max_abs_difference"] for r in REPRO),
                      "missing_cells": missing}, indent=1))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):    # as run_candidate_training / locked_assessment fitted and scored
        run()
