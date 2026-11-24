"""Metrics of the set-count ONA runs (run_ona_setcount.py), next to single-judgment ONA, the
additive NARS/LLR control and LR on the judgments. Laptop side, light: reads synced results only.

Every set-count score is the expectation of ONA's printed answer for the pair's episode
(experiments/ona/setcount/runs/<design>/answers.csv.gz). Before any metric, each pair's episode
input is rebuilt here from the laptop's knowledge and must hash to the input the desktop ran, and
every answer line is re-parsed from the archived NAR output. A pair without an answer keeps NaN and
its cell is not computed (reported), never filled.

Protocol (as run_ona_metrics.py / run_judgment_lr.py): thresholds by
run_experiments.threshold_on_validation on the selection role (highest tied cutoff); test scored
once; the second (locked) assessment uses the development thresholds of the same engine and
training regime; partitions by publication_evaluation.full_node_clusters over the reference
node lists. The ONA and LLR rows recomputed here must equal the saved tables.
Movers: true matches whose commune (hreppur, judgment different_county) and county (sysla,
different_district) both differ (paper-review/round_o2/03_domain_experts.md item 1).
"""
from __future__ import annotations

import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "vendor/bench"))

import ona_matcher as om  # noqa: E402
import ona_cache as oc  # noqa: E402
import run_ona_setcount as sc  # noqa: E402
from run_experiments import metrics, threshold_on_validation  # noqa: E402
from publication_evaluation import full_node_clusters  # noqa: E402
from run_ona_full import ICE, PUB, LOCKED, SEEDS  # noqa: E402

OUT = sc.OUT
TABLES = OUT / "tables"
SINGLE = {"ona": "ona_score", "published_nars": "llr_score"}
POPULATIONS = [("controlled", "controlled", "controlled", "pub"),                # (population, stem, training, kb)
               ("publication_graph", "publication_graph", "controlled", "pub"),
               ("candidate_graph", "candidate_graph", "candidate_matched", "cand"),
               ("locked_controlled", "locked_controlled", "controlled", "pub"),
               ("locked_candidate", "locked_candidate", "candidate_matched", "cand")]
MOVER = {"different_district", "different_county"}


class Missing(Exception):
    pass


def verified_answers(design: str, knowledge: dict, registry: dict) -> pd.DataFrame:
    """answers.csv.gz, with every input hash rebuilt and every answer line re-parsed from the archive."""
    root = OUT / f"runs/{design}"
    answers = pd.read_csv(root / "answers.csv.gz", keep_default_na=False, na_values=[""])
    store = sc.Store(root)
    lines_by_id = {}
    for path in sorted((root / "batches").glob("*.out.gz")):
        with gzip.open(path, "rt") as stream:
            for key, segment in oc.segments(stream.read(), oc.EP).items():
                if key is not None:
                    lines_by_id[int(key)] = segment.rstrip("\n").splitlines()[-1]
    refs, kinds = [], []
    for row in answers.itertuples(index=False):
        lines, reference = sc.episode_for(design, registry[row.judgment_set], row.judgment_set,
                                          knowledge[(row.seed, row.kb)])
        if oc.sha("\n".join(lines) + "\n") != row.input_sha256:
            raise RuntimeError(f"{design}: rebuilt input differs from the one ONA ran ({row.seed}/{row.kb}/{row.judgment_set})")
        record = store.get(row.input_sha256)
        if lines_by_id[record["episode_id"]] != row.answer_line or record["answer_line"] != row.answer_line:
            raise RuntimeError(f"{design}: archived answer differs for episode {record['episode_id']}")
        answer = oc.parse_answer(row.answer_line + "\n", oc.TERM)
        if (answer is None) != (not row.answered) or (answer and abs(answer["expectation"] - row.expectation) > 1e-12):
            raise RuntimeError(f"{design}: re-parsed answer differs for episode {record['episode_id']}")
        refs.append(reference)
        kinds.append(sc.classify(answer["stamp"], sc.input_roles(lines)) if answer else "unanswered")
    if kinds != answers.answer_kind.tolist():
        raise RuntimeError(f"{design}: answer kinds differ on re-classification")
    # Fidelity against the clean NAL derivation; for choice designs any exactly tied conclusion counts.
    devs, sources = [], []
    for row, ref in zip(answers.itertuples(index=False), refs):
        if not row.answered or ref is None:
            devs.append(np.nan)
            sources.append("none")
            continue
        options = ref.get("tied") or [ref]
        best = min(options, key=lambda o: abs(o["frequency"] - row.frequency) + abs(o["confidence"] - row.confidence))
        devs.append(max(abs(best["frequency"] - row.frequency), abs(best["confidence"] - row.confidence)))
        sources.append(best.get("source", "revision"))
    answers["dev_reference"] = devs
    answers["reference_source"] = sources
    answers["reference_tied"] = [len(r.get("tied", [])) > 1 if r else False for r in refs]
    if design == "revision":
        shares = []
        for row in answers.itertuples(index=False):
            k, name = knowledge[(row.seed, row.kb)], sc.set_atom(row.judgment_set)
            weights = {}
            for j in sorted(set(registry[row.judgment_set]) | {name}):
                if j in k:
                    _, c = sc.conclusion(k[j], sc.OBSERVATION_CONFIDENCE)
                    weights[j] = c / (1 - c)
            shares.append(weights.get(name, 0.0) / sum(weights.values()) if weights else np.nan)
        answers["set_evidence_share"] = shares
    return answers


def engine_scores(pairs: pd.DataFrame, answers: dict) -> pd.DataFrame:
    out = pairs.copy()
    for design, frame in answers.items():
        keyed = frame.set_index(["seed", "kb", "judgment_set"])
        idx = pd.MultiIndex.from_arrays([out.seed, out.kb, out.judgment_set])
        out[f"ona_set_{design}"] = keyed.expectation.reindex(idx).to_numpy()
        out[f"kind_{design}"] = keyed.answer_kind.reindex(idx).to_numpy()
        if design == "revision":
            out["set_evidence_share"] = keyed.set_evidence_share.reindex(idx).to_numpy()
    return out


def values(frame, column):
    v = frame[column].to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise Missing(f"{int((~np.isfinite(v)).sum())} pairs without an ONA answer")
    return v


def lr_movers(seed, sets, words, records):
    """LR on the judgments, per-pair scores of the graph test roles (run_judgment_lr's own functions)."""
    import run_judgment_lr as lj
    controlled = lj.pairs(seed, "controlled")
    train = lj.part(controlled, "train")
    model = lj.fit(seed, lj.design([sets[s] for s in train.judgment_set], words), train.label.to_numpy())
    graph = lj.pairs(seed, "publication_graph")
    g_sel, g_test = lj.part(graph, "selection"), lj.part(graph, "test")
    g_thr = threshold_on_validation(g_sel.label.to_numpy(), lj.score(model, sets, words, g_sel))
    pub = (g_test[["id1", "id2", "label", "judgment_set"]].assign(score=lj.score(model, sets, words, g_test)), g_thr)
    c_thr = json.loads((ICE / "judgment_lr/fit.json").read_text())["seeds"][str(seed)]["candidate_threshold"]
    cand = (pd.read_csv(ICE / f"judgment_lr/seed_{seed}/candidate_test.csv.gz", float_precision="round_trip"), c_thr)
    locked = (pd.read_csv(ICE / f"judgment_lr/seed_{seed}/locked_candidate.csv.gz", float_precision="round_trip"), c_thr)
    ctl = (pd.read_csv(ICE / f"judgment_lr/seed_{seed}/controlled_test.csv.gz", float_precision="round_trip"),
           json.loads((ICE / "judgment_lr/fit.json").read_text())["seeds"][str(seed)]["controlled_threshold"])
    return {"publication_graph": pub, "candidate_graph": cand, "locked_candidate": locked, "controlled": ctl}


def main():
    from resource_limits import check_memory
    check_memory(3)
    knowledge, registry = sc.load_knowledge(OUT), sc.load_registry(OUT)
    designs = [d for d in sc.DESIGNS if (OUT / f"runs/{d}/answers.csv.gz").exists()]
    answers = {d: verified_answers(d, knowledge, registry) for d in designs}
    pairs = pd.read_csv(OUT / "inputs/pairs.csv.gz", float_precision="round_trip")
    scored = engine_scores(pairs, answers)
    engines = {**{f"ona_set_{d}": f"ona_set_{d}" for d in designs}, **SINGLE}
    saved = {"controlled": pd.read_csv(ICE / "tables/metrics.csv", float_precision="round_trip"),
             "publication_graph": pd.read_csv(ICE / "tables/graph_metrics.csv", float_precision="round_trip"),
             "candidate_graph": pd.read_csv(ICE / "tables/candidate_graph_metrics.csv", float_precision="round_trip"),
             "locked": pd.read_csv(ICE / "tables/locked_metrics.csv", float_precision="round_trip")}
    lr = pd.read_csv(ICE / "tables/judgment_lr_metrics.csv", float_precision="round_trip")
    sets = {int(k): v for k, v in registry.items()}
    words = sorted({j for v in sets.values() for j in v})
    rows, movers, kinds, missing, checks = [], [], [], [], []
    for seed in SEEDS:
        state = {}
        lr_pairs = lr_movers(seed, sets, words, None)
        for population, stem, training, kb in POPULATIONS:
            frame = scored[scored.seed.eq(seed) & scored.stem.eq(stem)]
            locked = population.startswith("locked")
            test = frame[frame.partition.eq("locked" if locked else "test")].reset_index(drop=True)
            sel = None if locked else frame[frame.partition.eq("selection")].reset_index(drop=True)
            graph = "graph" in population or population == "locked_candidate"
            nodes = (pd.read_csv(LOCKED / "candidate_nodes.csv") if population == "locked_candidate" else
                     pd.read_csv(PUB / f"seed_{seed}/test_graph_nodes.csv")).set_index("id", drop=False) if graph else None
            mover = test.judgment_set.map(lambda s: MOVER <= sets[int(s)]).to_numpy() & test.label.eq(1).to_numpy()
            for engine, column in engines.items():
                try:
                    t_scores = values(test, column)
                    if locked:
                        key = (engine, "controlled" if population == "locked_controlled" else "candidate_graph")
                        threshold = state[key]
                    else:
                        threshold = threshold_on_validation(sel.label.to_numpy(), values(sel, column))
                        state[(engine, population)] = threshold
                except (Missing, KeyError) as error:
                    missing.append({"seed": seed, "population": population, "engine": engine, "reason": str(error)})
                    continue
                row = {"seed": seed, "population": population, "training": training, "model": engine,
                       **metrics(test.label.to_numpy(), t_scores, threshold)}
                if graph:
                    cluster, _ = full_node_clusters(nodes, test, t_scores, threshold)
                    row.update({k: cluster[k] for k in ["ari", "b3_f1", "n_predicted_clusters", "largest_component"]})
                rows.append(row)
                movers.append({"seed": seed, "population": population, "model": engine, "movers": int(mover.sum()),
                               "positives": int(test.label.sum()),
                               "mover_recall": float(np.mean(t_scores[mover] >= threshold)) if mover.any() else None,
                               "non_mover_recall": float(np.mean(t_scores[~mover & test.label.eq(1).to_numpy()] >= threshold))})
                if engine in SINGLE:      # recomputed single-judgment ONA and LLR rows must equal the saved tables
                    if population == "controlled":
                        ref = saved["controlled"].query("seed == @seed and model == @engine and evaluation == 'controlled_crosswave_reference'")
                    elif population == "publication_graph":
                        ref = saved["publication_graph"].query("seed == @seed and model == @engine")
                    elif population == "candidate_graph":
                        ref = saved["candidate_graph"].query("seed == @seed and model == @engine")
                    else:
                        pop = "controlled" if population == "locked_controlled" else "candidate"
                        ref = saved["locked"].query("seed == @seed and model == @engine and population == @pop")
                    if len(ref) != 1:
                        raise RuntimeError(f"No unique saved row for {seed}/{population}/{engine}")
                    ref = ref.iloc[0]
                    diff = max(abs(float(ref[k]) - float(row[k])) for k in ["f1", "precision", "recall", "auc", "threshold"]
                               + (["ari"] if graph else []))
                    checks.append({"seed": seed, "population": population, "model": engine, "max_abs_difference": diff})
                    if diff > 1e-9:
                        raise RuntimeError(f"Recomputed {engine} row differs from the saved table: {checks[-1]}")
                for design in designs:
                    if engine == f"ona_set_{design}":
                        counts = test[f"kind_{design}"].value_counts()
                        seen = test.judgment_set.map(lambda s, k=knowledge[(seed, kb)]: sc.set_atom(s) in k).to_numpy()
                        entry = {"seed": seed, "population": population, "design": design, "pairs": len(test),
                                 "seen_set_share": float(seen.mean()),
                                 **{f"answer_{k}": int(v) for k, v in counts.items()}}
                        if design == "revision":
                            entry["set_evidence_share_mean_seen"] = float(test.set_evidence_share[seen].mean())
                            entry["set_evidence_share_max"] = float(test.set_evidence_share.max())
                        kinds.append(entry)
            # LR on the judgments: saved rows; movers from its per-pair scores.
            lr_pop = {"controlled": ("controlled", "controlled"), "publication_graph": ("candidate_graph", "controlled"),
                      "candidate_graph": ("candidate_graph", "candidate_matched"),
                      "locked_controlled": ("locked_controlled", "controlled"),
                      "locked_candidate": ("locked_candidate", "candidate_matched")}[population]
            ref = lr.query("seed == @seed and population == @lr_pop[0] and training == @lr_pop[1]").iloc[0].to_dict()
            rows.append({"seed": seed, "population": population, "training": training, "model": "lr_judgments",
                         **{k: ref[k] for k in ["precision", "recall", "f1", "auc", "average_precision", "threshold", "tp", "fp",
                                                "tn", "fn", "n_pairs", "n_positive"]},
                         **({k: ref[k] for k in ["ari", "b3_f1", "n_predicted_clusters", "largest_component"]} if graph else {})})
            if population in lr_pairs:
                lp, lthr = lr_pairs[population]
                if not lp[["id1", "id2", "label"]].reset_index(drop=True).equals(test[["id1", "id2", "label"]]):
                    raise RuntimeError(f"LR per-pair file differs in cases for {seed}/{population}")
                lf1 = metrics(lp.label.to_numpy(), lp.score.to_numpy(), lthr)["f1"]
                if abs(lf1 - ref["f1"]) > 1e-9:
                    raise RuntimeError(f"LR per-pair scores do not reproduce the saved F1 for {seed}/{population}")
                s = lp.score.to_numpy()
                movers.append({"seed": seed, "population": population, "model": "lr_judgments", "movers": int(mover.sum()),
                               "positives": int(test.label.sum()),
                               "mover_recall": float(np.mean(s[mover] >= lthr)) if mover.any() else None,
                               "non_mover_recall": float(np.mean(s[~mover & test.label.eq(1).to_numpy()] >= lthr))})
        print(f"seed {seed} done", flush=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame, mv, kd = pd.DataFrame(rows), pd.DataFrame(movers), pd.DataFrame(kinds)
    frame.to_csv(TABLES / "metrics.csv", index=False)
    mv.to_csv(TABLES / "movers.csv", index=False)
    kd.to_csv(TABLES / "answer_kinds.csv", index=False)
    episodes = []
    for design, a in answers.items():
        a.to_csv(TABLES / f"episodes_{design}.csv.gz", index=False)
        run = json.loads((OUT / f"runs/{design}/run.json").read_text())
        fresh = OUT / f"runs/{design}/fresh_check/report.json"
        episodes.append({"design": design, "episodes": len(a), "answered": int(a.answered.sum()),
                         "unanswered": int((~a.answered.astype(bool)).sum()),
                         "max_dev_reference": float(np.nanmax(a.dev_reference)),
                         "dev_gt_1e-5": int((a.dev_reference > 1e-5).sum()), "tied_references": int(a.reference_tied.sum()),
                         "answer_kinds": a.answer_kind.value_counts().to_dict(),
                         "ona_seconds": sum(h["ona_seconds"] for h in run["history"]),
                         "workers": run["workers"], "fresh_check": json.loads(fresh.read_text()) if fresh.exists() else None})
    dump = sc.dump
    dump(TABLES / "episodes_summary.json", episodes)
    dump(TABLES / "checks.json", {"single_engine_rows_equal_saved": checks, "missing_cells": missing,
                                  "verified": "inputs rebuilt and hashed; answer lines re-parsed from the NAR archive",
                                  "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(json.dumps({"episodes": episodes, "missing": missing}, indent=1, default=str))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
