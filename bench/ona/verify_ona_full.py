"""Independent checks of the ONA full run (adversarial review; writes experiments/ona/full/verification.json).

1. Scores come from ONA output: every episode's answer line is re-read from the archived NAR
   transcripts (batches/*.out.gz), re-parsed here and its expectation recomputed; every
   per-pair ona_score must equal the expectation of the episode it maps to.
2. Episode inputs: each archived batch input, split at *reset, hashes (marker normalised) to
   the cache key of the episode it delimits.
3. Metrics recomputed from the per-pair files with separate code (plain numpy confusion counts,
   rank-sum AUROC, union-find components + sklearn ARI), compared with the ONA table rows.
4. Thresholds: every ONA threshold is re-derived from selection/validation scores alone (the
   locked thresholds must equal the development ones).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "experiments/ona/full"
ICE = OUT / "iceid"
ANSWER = re.compile(r"Answer: <\{px\} --> match>\. creationTime=-?\d+ Stamp=\[[\d,]*\] "
                    r"Truth: frequency=(\d+\.\d+), confidence=(\d+\.\d+)")


def transcripts(domain):
    """{episode_id: (answer line, input sha)} read straight from the archived NAR input/output."""
    found = {}
    for out_path in sorted((OUT / f"episodes/{domain}/batches").glob("run*.out.gz")):
        text = gzip.open(out_path, "rt").read()
        nal = gzip.open(str(out_path).replace(".out.gz", ".nal.gz"), "rt").read()
        current = None
        for line in text.splitlines():
            if line.startswith("Comment: ONA_EP|"):
                current = int(line.split("|", 1)[1])
            elif line.startswith("Answer:") and current is not None:
                found[current] = [line, None]
                current = None
        for block in nal.split("*reset\n")[1:]:
            block = "*reset\n" + block
            lines = block.rstrip("\n").split("\n")
            eid = int(lines[-2].split("|", 1)[1])
            canonical = "\n".join(lines[:-2] + ["//ONA_PAIR|x"] + lines[-1:]) + "\n"
            found[eid][1] = hashlib.sha256(canonical.encode()).hexdigest()
    return found


def expectation(line):
    if line == "Answer: None.":
        return float("nan")
    m = ANSWER.fullmatch(line)
    if m is None:
        raise RuntimeError("unparsed " + line)
    f, c = float(m.group(1)), float(m.group(2))
    return c * (f - .5) + .5


def confusion(y, s, t):
    p = s >= t
    tp, fp = int(np.sum(p & (y == 1))), int(np.sum(p & (y == 0)))
    fn, tn = int(np.sum(~p & (y == 1))), int(np.sum(~p & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def rank_auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def components(ids, edges):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.asarray([find(i) for i in ids])


def check_row(kind, row, y, s, out, mismatches):
    mine = confusion(y, s, row["threshold"])
    mine["auc"] = rank_auc(y, s) if 0 < y.sum() < len(y) else None
    for k, v in mine.items():
        if v is None:
            continue
        if abs(float(row[k]) - v) > 1e-9:
            mismatches.append({"row": kind, "field": k, "table": row[k], "independent": v})
    out.append(kind)


def main():
    report = {}
    for domain in ["iceid", "external"]:
        episodes = pd.read_csv(OUT / f"episodes/{domain}/episodes.csv.gz", float_precision="round_trip")
        found = transcripts(domain)
        bad_score = bad_hash = 0
        for row in episodes.itertuples():
            line, digest = found[row.episode_id]
            e = expectation(line)
            if not (np.isnan(e) and np.isnan(row.expectation)) and abs(e - row.expectation) > 0:
                bad_score += 1
            if line != row.answer_line:
                bad_score += 1
            if digest != row.input_sha256:
                bad_hash += 1
        report[f"{domain}_episodes"] = {"episodes": len(episodes), "transcript_answers": len(found),
                                        "expectation_or_line_mismatch": bad_score, "input_hash_mismatch": bad_hash}
    # Per-pair ona_score equals its episode's parsed expectation.
    ep = {d: pd.read_csv(OUT / f"episodes/{d}/episodes.csv.gz", float_precision="round_trip").set_index("episode_id").expectation for d in ["iceid", "external"]}
    pair_bad, pair_total = 0, 0
    for path in list((ICE / "pairs").glob("seed_*/*.csv.gz")) + list((OUT / "external").glob("*/*_scores.csv.gz")):
        f = pd.read_csv(path, usecols=["episode_id", "ona_score"], float_precision="round_trip")
        expected = ep["external" if "external" in str(path) else "iceid"].loc[f.episode_id].to_numpy()
        pair_bad += int(np.sum(~np.isclose(expected, f.ona_score.to_numpy(), rtol=0, atol=0, equal_nan=True)))
        pair_total += len(f)
    report["pair_scores_from_episodes"] = {"pairs": pair_total, "mismatches": pair_bad}
    # Independent metric recomputation and threshold provenance.
    sys.path.insert(0, str(BASE))
    from run_experiments import threshold_on_validation
    from sklearn.metrics import adjusted_rand_score
    T = ICE / "tables"
    done, mismatches, thresholds = [], [], []
    m = pd.read_csv(T / "metrics.csv", float_precision="round_trip")
    for seed in [131, 132, 133]:
        c = pd.read_csv(ICE / f"pairs/seed_{seed}/controlled.csv.gz", float_precision="round_trip")
        sel, test = c[c.partition.eq("selection")], c[c.partition.eq("test")]
        row = m[(m.seed == seed) & m.model.eq("ona") & m.evaluation.eq("controlled_crosswave_reference")].iloc[0]
        t = threshold_on_validation(sel.label.to_numpy(), sel.ona_score.to_numpy())
        thresholds.append({"seed": seed, "table": row.threshold, "from_selection_only": t, "equal": bool(row.threshold == t),
                           "population": "controlled"})
        check_row(f"controlled/{seed}", row, test.label.to_numpy(), test.ona_score.to_numpy(), done, mismatches)
        g = pd.read_csv(ICE / f"pairs/seed_{seed}/candidate_graph.csv.gz", float_precision="round_trip")
        gm = pd.read_csv(T / "candidate_graph_metrics.csv", float_precision="round_trip")
        row = gm[(gm.seed == seed) & gm.model.eq("ona")].iloc[0]
        gsel, gtest = g[g.partition.eq("selection")], g[g.partition.eq("test")]
        t = threshold_on_validation(gsel.label.to_numpy(), gsel.ona_score.to_numpy())
        thresholds.append({"seed": seed, "table": row.threshold, "from_selection_only": t, "equal": bool(row.threshold == t),
                           "population": "candidate_graph"})
        check_row(f"candidate_graph/{seed}", row, gtest.label.to_numpy(), gtest.ona_score.to_numpy(), done, mismatches)
        nodes = pd.read_csv(BASE / f"experiments/publication-quarantined-extended/seed_{seed}/test_graph_nodes.csv")
        accepted = gtest[gtest.ona_score.to_numpy() >= row.threshold]
        pred = components(nodes.id.tolist(), zip(accepted.id1, accepted.id2))
        ari = adjusted_rand_score(nodes.person.to_numpy(), pred)
        if abs(ari - row.ari) > 1e-12:
            mismatches.append({"row": f"candidate_graph/{seed}", "field": "ari", "table": row.ari, "independent": ari})
        ab = pd.read_csv(T / "ablations.csv", float_precision="round_trip")
        for cat in ["name", "geography", "birthyear", "sex", "wave"]:
            f = pd.read_csv(ICE / f"pairs/seed_{seed}/ablate_{cat}.csv.gz", float_precision="round_trip")
            row = ab[(ab.seed == seed) & ab.model.eq(f"ona_ablate_{cat}")].iloc[0]
            t = threshold_on_validation(f[f.partition.eq("selection")].label.to_numpy(), f[f.partition.eq("selection")].ona_score.to_numpy())
            thresholds.append({"seed": seed, "table": row.threshold, "from_selection_only": t, "equal": bool(row.threshold == t),
                               "population": f"ablate_{cat}"})
            ft = f[f.partition.eq("test")]
            check_row(f"ablate_{cat}/{seed}", row, ft.label.to_numpy(), ft.ona_score.to_numpy(), done, mismatches)
        lk = pd.read_csv(T / "locked_metrics.csv", float_precision="round_trip")
        for pop in ["controlled", "candidate"]:
            f = pd.read_csv(ICE / f"pairs/seed_{seed}/locked_{pop}.csv.gz", float_precision="round_trip")
            row = lk[(lk.seed == seed) & lk.model.eq("ona") & lk.population.eq(pop)].iloc[0]
            dev = (m[(m.seed == seed) & m.model.eq("ona") & m.evaluation.eq("controlled_crosswave_reference")].iloc[0].threshold
                   if pop == "controlled" else gm[(gm.seed == seed) & gm.model.eq("ona")].iloc[0].threshold)
            thresholds.append({"seed": seed, "table": row.threshold, "from_selection_only": dev, "equal": bool(row.threshold == dev),
                               "population": f"locked_{pop}"})
            check_row(f"locked_{pop}/{seed}", row, f.label.to_numpy(), f.ona_score.to_numpy(), done, mismatches)
    ext = pd.read_csv(OUT / "external/results.csv", float_precision="round_trip")
    sys.path.insert(0, str(BASE / "vendor/bench"))
    import ast
    tree = ast.parse((BASE / "vendor/bench/scripts/run_nars_full_eval.py").read_text())
    ns = {"np": np}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_threshold_by_f1"],
                            type_ignores=[]), "sel", "exec"), ns)
    for d in ext.dataset.unique():
        r = ext[ext.dataset.eq(d) & ext.model.eq("ona")]
        if not len(r) or pd.isna(r.iloc[0].get("f1")):
            continue
        r = r.iloc[0]
        v = pd.read_csv(OUT / f"external/{d}/valid_scores.csv.gz", float_precision="round_trip")
        t = pd.read_csv(OUT / f"external/{d}/test_scores.csv.gz", float_precision="round_trip")
        th = float(ns["select_threshold_by_f1"](v.ona_score.to_numpy(), v.label.to_numpy(), default=0.5))
        thresholds.append({"dataset": d, "table": r.threshold, "from_validation_only": th, "equal": bool(abs(round(th, 4) - r.threshold) < 1e-12)})
        mine = confusion(t.label.to_numpy(), t.ona_score.to_numpy(), th)
        for k in ["precision", "recall", "f1"]:
            if abs(round(mine[k], 4) - r[k]) > 1e-12:
                mismatches.append({"row": f"external/{d}", "field": k, "table": r[k], "independent": mine[k]})
        auc = rank_auc(t.label.to_numpy(), t.ona_score.to_numpy())      # tie-averaged
        if abs(auc - r.auc_tie_averaged) > 1e-12:
            mismatches.append({"row": f"external/{d}", "field": "auc_tie_averaged", "table": r.auc_tie_averaged, "independent": auc})
        # The evaluator's own AUC: ranks by stable sort order, ties not averaged.
        order = np.argsort(t.ona_score.to_numpy(), kind="mergesort")
        ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
        y = t.label.to_numpy(); n1 = int(y.sum())
        bench_auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
        if abs(round(bench_auc, 4) - r.auc) > 1e-12:
            mismatches.append({"row": f"external/{d}", "field": "auc", "table": r.auc, "independent": bench_auc})
        done.append(f"external/{d}")
    report["independent_metrics"] = {"rows_checked": len(done), "mismatches": mismatches}
    report["thresholds"] = {"checked": len(thresholds), "all_equal": all(x["equal"] for x in thresholds), "rows": thresholds}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: (v if k != "thresholds" else {"checked": v["checked"], "all_equal": v["all_equal"]})
                      for k, v in report.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
