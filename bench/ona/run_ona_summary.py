"""summary.csv and summary.md: ONA versus the additive NARS/LLR control for every replaced cell.

Reads only the outputs of run_ona_full.py, run_ona_metrics.py and run_ona_external.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from run_ona_full import OUT, ICE  # noqa: E402

T = ICE / "tables"
NAMES = {"name": "names", "geography": "geography", "birthyear": "birth year", "sex": "sex", "wave": "census wave"}
ROWS = []


def cell(table, row, metric, ona, llr, kind="mean"):
    ROWS.append({"table": table, "row": row, "metric": metric, "statistic": kind,
                 "ona": ona, "llr": llr, "delta_ona_minus_llr": (ona - llr) if isinstance(ona, (int, float)) and
                 isinstance(llr, (int, float)) and np.isfinite(ona) and np.isfinite(llr) else None})


def pair_stats(frame, metric, ona_mask, llr_mask, table, row, sd=True):
    o, l = frame.loc[ona_mask, metric], frame.loc[llr_mask, metric]
    if len(o) == 0:
        cell(table, row, metric, float("nan"), float(l.mean()))
        return
    cell(table, row, metric, float(o.mean()), float(l.mean()))
    if sd and len(o) > 1:
        cell(table, row, metric, float(o.std()), float(l.std()), "sd")


def fmt(v, digits=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def build():
    m = pd.read_csv(T / "metrics.csv")
    for ev, label in [("controlled_crosswave_reference", "Controlled pairs (test 3,600 x 3 seeds)"),
                      ("audit_provisional_reference", "Audit provisional reference (136)"),
                      ("audit_inherited_reference", "Audit inherited reference (136)")]:
        g = m[m.evaluation.eq(ev)]
        for metric in (["precision", "recall", "f1", "auc", "average_precision", "threshold"] if ev.startswith("controlled")
                       else ["f1", "provisional_different_linked"]):
            pair_stats(g, metric, g.model.eq("ona"), g.model.eq("published_nars"), "controlled_and_audit", label, metric == "f1")
    a = pd.read_csv(T / "audit_source_timing.csv")
    for pop in ["full_provisional", "exclude_nominal1816", "exclude_source_uncertain_jon"]:
        g = a[a.population.eq(pop)]
        for metric in ["f1", "provisional_different_linked"]:
            pair_stats(g, metric, g.model.eq("ona"), g.model.eq("published_nars"), "audit_source_timing", pop, False)
    g = pd.read_csv(T / "graph_metrics.csv")
    for metric in ["f1", "average_precision", "b3_precision", "b3_recall", "b3_f1", "ari", "partition_pair_f1"]:
        pair_stats(g, metric, g.model.eq("ona"), g.model.eq("published_nars"), "publication_trained_graph", "Controlled-trained, candidate graph test", metric in ["f1", "b3_f1"])
    c = pd.read_csv(T / "candidate_graph_metrics.csv")
    for metric in ["f1", "average_precision", "b3_precision", "b3_recall", "b3_f1", "ari", "partition_pair_f1", "largest_component", "auc"]:
        pair_stats(c, metric, c.model.eq("ona"), c.model.eq("published_nars"), "candidate_graph", "Candidate-matched training (Table: graph)", metric in ["f1", "b3_f1", "partition_pair_f1"])
    r = pd.read_csv(T / "candidate_ranking.csv")
    pair_stats(r, "p_at_k", r.model.eq("ona"), r.model.eq("published_nars"), "candidate_graph", "Candidate-matched training (Table: graph)", False)
    ab = pd.read_csv(T / "ablations.csv")
    full = m[m.evaluation.eq("controlled_crosswave_reference")]
    for cat, label in NAMES.items():
        o, l = ab[ab.model.eq(f"ona_ablate_{cat}")], ab[ab.model.eq(f"ablate_{cat}")]
        for metric in ["f1", "auc"]:
            pair_stats(ab, metric, ab.model.eq(f"ona_ablate_{cat}"), ab.model.eq(f"ablate_{cat}"), "ablations", "Without " + label, metric == "f1")
        cell("ablations", "Without " + label, "delta_f1_vs_full", float(o.f1.mean() - full[full.model.eq("ona")].f1.mean()) if len(o) else float("nan"),
             float(l.f1.mean() - full[full.model.eq("published_nars")].f1.mean()))
    cal = pd.read_csv(T / "calibration_metrics.csv")
    cc = pd.read_csv(T / "candidate_calibration_metrics.csv")
    for pop, frame in [("Controlled pairs", cal[cal.population.eq("controlled_2to1")]), ("Candidate graph (candidate-trained)", cc),
                       ("Candidate graph (controlled-trained)", cal[cal.population.eq("candidate_graph")])]:
        for method in ["raw", "sigmoid", "isotonic"]:
            f = frame[frame.calibration.eq(method)]
            for metric in ["f1", "brier", "log_loss"]:
                pair_stats(f, metric, f.model.eq("ona"), f.model.eq("published_nars"), "calibration", f"{pop} / {method}", False)
    ts = pd.read_csv(T / "threshold_sensitivity.csv")
    for policy in ["fixed_0.5", "training_f1", "selection_f1"]:
        f = ts[ts.policy.eq(policy)]
        for metric in ["f1", "threshold"]:
            pair_stats(f, metric, f.model.eq("ona"), f.model.eq("published_nars"), "threshold_policy", policy, metric == "f1")
    lk = pd.read_csv(T / "locked_metrics.csv")
    for pop in ["controlled", "candidate"]:
        f = lk[lk.population.eq(pop)]
        for metric in (["f1", "auc", "brier", "log_loss"] + (["b3_f1", "ari", "p_at_k"] if pop == "candidate" else [])):
            pair_stats(f, metric, f.model.eq("ona"), f.model.eq("published_nars"), "locked_assessment", pop, metric in ["f1", "b3_f1"])
    lc = pd.read_csv(T / "locked_calibration_metrics.csv")
    for pop in ["controlled", "candidate"]:
        for method in ["sigmoid", "isotonic"]:
            f = lc[lc.population.eq(pop) & lc.calibration.eq(method)]
            for metric in ["brier", "log_loss", "f1"]:
                pair_stats(f, metric, f.model.eq("ona"), f.model.eq("published_nars"), "locked_calibration", f"{pop} / {method}", False)
    boot = json.loads((T / "uncertainty.json").read_text())
    for name, b in boot.items():
        s = b["standard_reference_llr"]["models"]
        cell("component_bootstrap", name, "f1_interval_low", s["ona"]["f1_interval"][0], s["published_nars"]["f1_interval"][0], "2.5%")
        cell("component_bootstrap", name, "f1_interval_high", s["ona"]["f1_interval"][1], s["published_nars"]["f1_interval"][1], "97.5%")
        cell("component_bootstrap", name, "paired_delta_ona_minus_llr_low", s["ona"]["paired_delta_vs_llr_interval"][0], 0.0, "2.5%")
        cell("component_bootstrap", name, "paired_delta_ona_minus_llr_high", s["ona"]["paired_delta_vs_llr_interval"][1], 0.0, "97.5%")
    ext = pd.read_csv(OUT / "external/results.csv")
    for d in ext.dataset.unique():
        f = ext[ext.dataset.eq(d)]
        for metric in ["precision", "recall", "f1", "accuracy", "threshold", "auc", "auc_tie_averaged", "ari_cc", "ari_ag", "p_at_k", "r_at_k", "average_precision"]:
            o = f.loc[f.model.eq("ona"), metric] if metric in f else pd.Series(dtype=float)
            l = f.loc[f.model.eq("published_nars"), metric]
            cell("external_official_splits", d, metric, float(o.iloc[0]) if len(o) and pd.notna(o.iloc[0]) else float("nan"), float(l.iloc[0]))
    frame = pd.DataFrame(ROWS)
    frame.to_csv(OUT / "summary.csv", index=False)
    return frame, ext


def anomaly_counts():
    rows = []
    for domain in ["iceid", "external"]:
        e = pd.read_csv(OUT / f"episodes/{domain}/episodes.csv.gz")
        rows.append({"domain": domain, "distinct_episodes": len(e), "answered": int(e.answered.sum()),
                     "unanswered": int((~e.answered).sum()),
                     "deviation_gt_1e-5": int(e.anomaly.fillna("").eq("deviation").sum()),
                     "max_abs_dev_expectation": float(e.dev_expectation.max()),
                     "max_abs_dev_frequency": float(e.dev_frequency.max()),
                     "max_abs_dev_confidence": float(e.dev_confidence.max())})
    pair_rows = []
    for path in sorted((ICE / "pairs").glob("seed_*/*.csv.gz")):
        f = pd.read_csv(path, usecols=["answered", "anomaly"])
        pair_rows.append({"file": str(path.relative_to(OUT)), "pairs": len(f), "unanswered": int((~f.answered).sum()),
                          "anomalous": int(f.anomaly.fillna("").ne("").sum())})
    for path in sorted((OUT / "external").glob("*/*_scores.csv.gz")):
        f = pd.read_csv(path, usecols=["answered", "anomaly"])
        pair_rows.append({"file": str(path.relative_to(OUT)), "pairs": len(f), "unanswered": int((~f.answered).sum()),
                          "anomalous": int(f.anomaly.fillna("").ne("").sum())})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def ms(frame, table, row, metric, digits=3):
    """'mean ± SD' strings for ONA and LLR from summary rows."""
    f = frame[frame.table.eq(table) & frame.row.eq(row) & frame.metric.eq(metric)]
    mean, sd = f[f.statistic.eq("mean")], f[f.statistic.eq("sd")]
    if not len(mean):
        return "n/a", "n/a", "n/a"
    m = mean.iloc[0]
    o = fmt(m.ona, digits) + (f" ± {fmt(sd.iloc[0].ona, digits)}" if len(sd) else "")
    l = fmt(m.llr, digits) + (f" ± {fmt(sd.iloc[0].llr, digits)}" if len(sd) else "")
    d = fmt(m.delta_ona_minus_llr, digits) if m.delta_ona_minus_llr is not None and pd.notna(m.delta_ona_minus_llr) else "n/a"
    if d != "n/a" and not d.startswith("-"):
        d = "+" + d
    return o, l, d


def markdown(frame, ext, episodes, pairs):
    scoring = json.loads((ICE / "scoring.json").read_text())
    rt = scoring["runtime"]
    lines = ["# ONA replacement of the NARS/LLR matcher: full results", "",
             "Results only (no paper or existing file edited). ONA = OpenNARS for Applications does the inference "
             "(one reset episode per distinct input; score = expectation of ONA's printed answer). "
             "LLR = the additive log-likelihood-ratio control with the identical judgments and the identical "
             "training counts, recomputed here on the same pairs (every LLR cell reproduces the published value; "
             "`iceid/tables/llr_reproduction.json`). Values: mean ± sample SD over seeds 131-133 where the paper uses three seeds.", "",
             "## Runtime and settings", "",
             f"- ONA binary `{rt['binary']}`, sha256 `{rt['sha256']}`, commit `{rt['commit']}`, src clean: {rt['src_clean']}, mode {rt['mode']}",
             f"- Design (pilot, unchanged): {json.dumps(scoring['design'])}; implication truth f = w+/w (or 1-f for the "
             "negated form), c = w/(w+1) capped at 0.99; `*currenttime=1000000000` after each `*reset`; `*volume=0`.",
             f"- Episode cache: constant pair term `{scoring['cache']['pair_term']}`; ICE-ID {scoring['cache']['pairs_scored']:,} "
             f"pair scorings from {scoring['cache']['episodes']:,} distinct episodes ({scoring['cache']['judgment_sets']:,} distinct judgment sets).", ""]
    lines += ["## ICE-ID headline cells", ""]
    spec = [("controlled_and_audit", "Controlled pairs (test 3,600 x 3 seeds)", m, lab) for m, lab in
            [("precision", "Controlled P"), ("recall", "Controlled R"), ("f1", "Controlled F1"), ("auc", "Controlled AUROC"),
             ("average_precision", "Controlled AP"), ("threshold", "Controlled threshold (selection F1)")]]
    spec += [("controlled_and_audit", "Audit inherited reference (136)", "f1", "Audit inherited F1"),
             ("controlled_and_audit", "Audit provisional reference (136)", "f1", "Audit provisional F1"),
             ("controlled_and_audit", "Audit provisional reference (136)", "provisional_different_linked", "Audit different linked (of 117)"),
             ("audit_source_timing", "exclude_nominal1816", "f1", "Audit without 1816 F1"),
             ("audit_source_timing", "exclude_nominal1816", "provisional_different_linked", "Audit without 1816 different linked (of 113)"),
             ("audit_source_timing", "exclude_source_uncertain_jon", "f1", "Audit source-qualified 135 F1"),
             ("audit_source_timing", "exclude_source_uncertain_jon", "provisional_different_linked", "Audit source-qualified different linked (of 116)")]
    G = "Candidate-matched training (Table: graph)"
    spec += [("candidate_graph", G, mm, "Candidate graph " + lab) for mm, lab in
             [("f1", "edge F1"), ("average_precision", "AP"), ("b3_precision", "B3 P"), ("b3_recall", "B3 R"), ("b3_f1", "B3 F1"),
              ("ari", "ARI"), ("partition_pair_f1", "partition pair F1"), ("p_at_k", "P@k"), ("largest_component", "largest cluster")]]
    P = "Controlled-trained, candidate graph test"
    spec += [("publication_trained_graph", P, mm, "Controlled-trained graph " + lab) for mm, lab in
             [("f1", "edge F1"), ("b3_f1", "B3 F1"), ("ari", "ARI")]]
    spec += [("locked_assessment", "controlled", "f1", "Locked controlled F1"), ("locked_assessment", "candidate", "f1", "Locked edge F1"),
             ("locked_assessment", "candidate", "b3_f1", "Locked B3 F1"), ("locked_assessment", "candidate", "ari", "Locked ARI")]
    rows = [[lab, *ms(frame, t, r, mm, 0 if mm in ["provisional_different_linked", "largest_component"] else 4 if mm == "threshold" else 3)]
            for t, r, mm, lab in spec]
    lines += [md_table(["Cell", "ONA", "NARS/LLR (additive control)", "ONA - LLR"], rows), ""]
    lines += ["## Ablations (controlled test; counts re-fitted on train without the category, threshold re-selected)", ""]
    rows = []
    for name in ["names", "geography", "birth year", "sex", "census wave"]:
        o, l, d = ms(frame, "ablations", "Without " + name, "f1")
        dv = frame[frame.table.eq("ablations") & frame.row.eq("Without " + name) & frame.metric.eq("delta_f1_vs_full")].iloc[0]
        ao, al, _ = ms(frame, "ablations", "Without " + name, "auc")
        rows.append(["Without " + name, o, fmt(dv.ona), ao, l, fmt(dv.llr), al])
    lines += [md_table(["Vocabulary", "ONA F1", "ONA dF1 vs full", "ONA AUROC", "LLR F1", "LLR dF1", "LLR AUROC"], rows), ""]
    lines += ["## Calibration (maps fit on the calibration partition, threshold on selection)", ""]
    rows = []
    for pop in ["Controlled pairs", "Candidate graph (candidate-trained)", "Candidate graph (controlled-trained)"]:
        for method in ["raw", "sigmoid", "isotonic"]:
            r = f"{pop} / {method}"
            rows.append([pop, method] + [ms(frame, "calibration", r, mm, 4 if mm != "f1" else 3)[i] for mm in ["f1", "brier", "log_loss"] for i in [0, 1]])
    lines += [md_table(["Population", "Map", "ONA F1", "LLR F1", "ONA Brier", "LLR Brier", "ONA log loss", "LLR log loss"], rows), ""]
    lines += ["## Locked (fixed-model) calibration", ""]
    rows = []
    for pop in ["controlled", "candidate"]:
        rows.append([pop, "raw", *[ms(frame, "locked_assessment", pop, mm, 4)[i] for mm in ["brier", "log_loss"] for i in [0, 1]]])
        for method in ["sigmoid", "isotonic"]:
            rows.append([pop, method, *[ms(frame, "locked_calibration", f"{pop} / {method}", mm, 4)[i] for mm in ["brier", "log_loss"] for i in [0, 1]]])
    lines += [md_table(["Population", "Map", "ONA Brier", "LLR Brier", "ONA log loss", "LLR log loss"], rows), ""]
    lines += ["## Threshold-policy sensitivity (controlled test)", ""]
    rows = [[pol, *[ms(frame, "threshold_policy", pol, mm, 3 if mm == "f1" else 4)[i] for mm in ["f1", "threshold"] for i in [0, 1]]]
            for pol in ["fixed_0.5", "training_f1", "selection_f1"]]
    lines += [md_table(["Policy", "ONA F1", "LLR F1", "ONA mean threshold", "LLR mean threshold"], rows), ""]
    boot = json.loads((T / "uncertainty.json").read_text())
    lines += ["## Component bootstrap (seed 131; 500 resamples, rng 571, identical resamples for all models)", ""]
    rows = []
    for name, b in boot.items():
        s, v = b["standard_reference_llr"], b["reference_ona"]["models"]
        rows.append([name, s["n_components"], "[{:.4f}, {:.4f}]".format(*s["models"]["ona"]["f1_interval"]),
                     "[{:.4f}, {:.4f}]".format(*s["models"]["published_nars"]["f1_interval"]),
                     "[{:+.4f}, {:+.4f}]".format(*s["models"]["ona"]["paired_delta_vs_llr_interval"])])
    lines += [md_table(["Population", "Components", "ONA F1 95%", "LLR F1 95%", "ONA - LLR paired 95%"], rows), "",
              "Every other model's interval and its delta versus LLR reproduce the saved uncertainty.json exactly; deltas versus ONA "
              "for every model are in `iceid/tables/uncertainty.json` (`reference_ona`).", ""]
    lines += ["## Table 5 (F1) and the external benchmarks on the official DeepMatcher splits", "",
              "External: counts from the official train split, threshold maximising validation F1 "
              "(run_nars_full_eval.select_threshold_by_f1), official test scored once, metrics by the executed NARS evaluator. "
              "Fodors-Zagats without `class` (run_fodors_official split; the LLR rerun reproduces experiments/completion/fodors-official/nars exactly). "
              "The archived nars_full_eval.csv external cells used unretained sampled pairs and are not comparable.", ""]
    ice = frame[frame.table.eq("controlled_and_audit") & frame.row.str.startswith("Controlled") & frame.metric.eq("f1") & frame.statistic.eq("mean")].iloc[0]
    rows = [["ICE-ID (controlled, 3 seeds)", fmt(ice.ona), fmt(ice.llr), fmt(ice.delta_ona_minus_llr)]]
    for d in ext.dataset.unique():
        f = frame[frame.table.eq("external_official_splits") & frame.row.eq(d) & frame.metric.eq("f1")].iloc[0]
        rows.append([d, fmt(f.ona, 4), fmt(f.llr, 4), fmt(f.delta_ona_minus_llr, 4)])
    lines += [md_table(["Dataset", "ONA F1", "NARS/LLR F1 (same split)", "ONA - LLR"], rows), ""]
    fields = ["precision", "recall", "f1", "accuracy", "threshold", "auc", "auc_tie_averaged", "ari_cc", "ari_ag", "p_at_k", "r_at_k"]
    lines += ["`auc` is the evaluator's own (bench _roc_auc_score: stable-sort ranks, ties not averaged, so it depends on pair order "
              "whenever scores tie, as they do for both matchers here); `auc_tie_averaged` is sklearn roc_auc_score on the same scores.", ""]
    for engine, label in [("ona", "ONA"), ("published_nars", "NARS/LLR")]:
        rows = []
        for d in ext.dataset.unique():
            r = ext[ext.dataset.eq(d) & ext.model.eq(engine)].iloc[0]
            rows.append([d] + [fmt(r.get(k), 4) if k in r else "n/a" for k in fields] +
                        [f"{int(r.n_test_pairs)}/{int(r.n_positives)}" if pd.notna(r.get("n_test_pairs")) else "n/a"])
        lines += [f"{label} (nars_all_benchmarks layout):", "", md_table(["Dataset"] + fields + ["test pairs/pos"], rows), ""]
    lines += ["## Anomalies (never replaced)", "",
              "Anomaly = unanswered episode, or |ONA - clean NAL derivation| > 1e-5 in frequency, confidence or expectation.", "",
              md_table(list(episodes.columns), episodes.values.tolist()), ""]
    agg = pairs.assign(domain=np.where(pairs.file.str.contains("external"), "external", "iceid")).groupby("domain")[["pairs", "unanswered", "anomalous"]].sum().reset_index()
    lines += ["Pair-level (each pair scored in each protocol counted once):", "", md_table(list(agg.columns), agg.values.tolist()), ""]
    missing = json.loads((T / "llr_reproduction.json").read_text())["missing_cells"]
    lines += [f"Cells not computed because of unanswered pairs: {len(missing)}" + (f" ({missing})" if missing else ""), ""]
    for name, path in [("Cache equivalence", OUT / "cache_check/report.json"), ("Concept-memory saturation", OUT / "saturation/summary.json"),
                       ("Independent verification", OUT / "verification.json")]:
        if path.exists():
            obj = json.loads(path.read_text())
            if name == "Independent verification":
                obj = {k: (v if k != "thresholds" else {"checked": v["checked"], "all_equal": v["all_equal"]}) for k, v in obj.items()}
            if name == "Cache equivalence":
                obj = {k: v for k, v in obj.items() if k != "runtime"}
            lines += [f"## {name}", "", "```json", json.dumps(obj, indent=1, default=str), "```", ""]
    ex = OUT / "explanations/explanations.json"
    if ex.exists():
        lines += ["## ONA per-pair explanations (seed 131, the pairs of the current figures)", ""]
        for e in json.loads(ex.read_text()):
            lines += [f"Records {e['id1']} / {e['id2']} (reference {e['reference_label']}); ONA: `{e['ona_answer_line']}`; "
                      f"identical to the scored run: {e['identical_to_scored_run']}; shares reproduce f "
                      f"(|diff| {e['abs_difference_frequency']:.1e}) and c (|diff| {e['abs_difference_confidence']:.1e}); "
                      f"expectation {e['ona_expectation']:.6f} (LLR raw score {e['llr_figure']['raw_score']:.6f}).", ""]
            rows = [[p["judgment"], p.get("w_plus", ""), p.get("w", ""), fmt(p.get("frequency"), 6), fmt(p.get("confidence"), 6),
                     fmt(p.get("evidence_weight"), 4), fmt(p.get("share"), 4), fmt(p.get("frequency_contribution"), 4)] for p in e["parts"]]
            lines += [md_table(["Judgment", "w+", "w", "conclusion f", "conclusion c", "w = c/(1-c)", "share", "share x f"], rows), ""]
    m = pd.read_csv(T / "metrics.csv", float_precision="round_trip")
    t131 = m[(m.seed == 131) & m.model.eq("ona") & m.evaluation.eq("controlled_crosswave_reference")].threshold.iloc[0]
    notes = ["## Notes for integration", "",
             "- ONA scores are coarse: every pair with the same judgment set gets the same answer, so scores tie heavily "
             "(ICE-ID 1,250 distinct judgment sets over 1.63M scorings; external 31-404 distinct episodes per dataset). "
             "The selection-F1 threshold rule and tie-averaged sklearn AUROC/AP handle ties; the external evaluator's `auc` does not (see above).",
             "- ONA expectations sit in a narrow band (controlled selection-F1 thresholds 0.256-0.292 versus 0.016-0.028 for LLR); "
             "the raw ONA expectation is not a calibrated probability (raw Brier/log loss worse than LLR; after sigmoid/isotonic maps they are close).",
             "- Concept memory (CONCEPTS_MAX 16,384) was full after 212 of the 213 diagnosed episodes with 10 or more judgments (none below 9 judgments); no answer in any domain deviates from "
             "the clean NAL derivation by more than 6.5e-7, so the saturation did not cost evidence here (`saturation/`).",
             "- The training-F1 threshold policy scores the training pairs with knowledge counted on those same pairs (in-sample), as the LLR row does.",
             f"- Explanation pairs are the current figures' pairs (chosen by the LLR criteria). Under ONA (seed-131 threshold {t131:.6f}) both are "
             "still linked: the reference-positive pair is a true positive and the reference-negative pair is still a false positive.",
             "- External NARS/LLR on the official splits differs strongly from the archived nars_full_eval.csv cells (e.g. Abt-Buy 0.356 vs 0.648, "
             "iTunes-Amazon 1.000 vs 0.757); the archived cells used unretained sampled pairs. Both ONA and LLR rows here use identical splits and counts.",
             "- Model names in the tables: `ona`, `ona_ablate_<category>`; LLR rows keep `published_nars` / `ablate_<category>` and equal the published rows.", ""]
    lines += notes
    timing = {"iceid_ona_seconds": scoring["ona_seconds_this_run"], "iceid_wall_seconds_incl_judgments": scoring["wall_seconds"],
              "external_ona_wall_seconds_by_dataset (batch archive timestamps)": {d: v["wall_seconds"] for d, v in json.loads((OUT / "timing.json").read_text())["external_by_dataset"].items()}}
    lines += ["## Runtime (seconds)", "", "```json", json.dumps(timing, indent=1), "```", ""]
    lines += ["## Reproduce", "", "```bash", "cd label-repair",
              "python source_alias.py run_ona_full.py iceid-inputs   # observed records, publication path",
              "python run_ona_full.py iceid                          # re-fit counts, judgments, LLR checks, ONA episodes",
              "python run_ona_metrics.py                             # all ICE-ID tables (LLR rows checked against the published ones)",
              "python run_ona_external.py                            # 8 official DeepMatcher splits, LLR + ONA",
              "python run_ona_full.py fresh-check                    # cached vs per-pair episodes",
              "python run_ona_full.py saturation", "python run_ona_full.py explain",
              "python verify_ona_full.py                             # independent re-parse / recomputation",
              "python run_ona_summary.py", "```", "",
              "## Files", "",
              "- `iceid/tables/*.csv`: same columns as the published files; ONA rows model `ona` / `ona_ablate_<category>`, "
              "LLR rows the original names (metrics, audit_source_timing, threshold_sensitivity, calibration_metrics, graph_metrics, "
              "ablations, candidate_graph_metrics, candidate_ranking, candidate_calibration_metrics, locked_metrics, "
              "locked_calibration_metrics, knowledge, uncertainty.json, llr_reproduction.json)",
              "- `iceid/pairs/seed_<s>/*.csv.gz`: per-pair id1, id2, label, knowledge base, episode_id, ona_score, llr_score, anomaly (the pair -> episode map)",
              "- `episodes/<domain>/episodes.csv.gz` (one row per distinct input: answer line, f, c, NAL reference, deviation, anomaly), "
              "`episodes/<domain>/batches/*.nal.gz|*.out.gz` (every NAR process input/output)",
              "- `external/<dataset>/{result.json,valid_scores.csv.gz,test_scores.csv.gz}`, `external/results.csv`",
              "- `cache_check/`, `saturation/`, `explanations/`, `verification.json`, `summary.csv`", ""]
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    frame, ext = build()
    episodes, pairs = anomaly_counts()
    episodes.to_csv(OUT / "anomalies_episodes.csv", index=False)
    pairs.to_csv(OUT / "anomalies_pairs.csv", index=False)
    markdown(frame, ext, episodes, pairs)
    print((OUT / "summary.md").read_text())
    print(episodes.to_string())
    print(pairs.groupby(pairs.file.str.contains("external")).sum(numeric_only=True))
