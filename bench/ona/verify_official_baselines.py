"""Independent check of experiments/completion/official-baselines (no code shared with the drivers).

For every <method>/<dataset>/test_predictions.csv: the rows must be the official DeepMatcher test.csv
read straight from the archive (ids, labels, order); P/R/F1 are recomputed with scikit-learn from the
saved decisions and compared with result.json and official_baselines.csv; the decision must equal
score >= threshold (Ditto: score > threshold, its evaluate() rule). AnyMatch cells must also use the
fixed 0.5 cutoff, the unique external-validation checkpoint, zero test-title overlap, and, on a DBLP
target, both DBLP domains left out. Writes verification.json.
"""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

BASE = Path(__file__).resolve().parent
ROOT = BASE / "experiments/completion/official-baselines"
ARCHIVES = {"abt_buy": "abt_buy_exp_data.zip", "amazon_google": "amazon_google_exp_data.zip",
            "dblp_acm": "dblp_acm_exp_data.zip", "dblp_scholar": "dblp_scholar_exp_data.zip",
            "itunes_amazon": "itunes_amazon_exp_data.zip", "walmart_amazon": "walmart_amazon_exp_data.zip",
            "beer": "beer_exp_data.zip"}
ANYMATCH_CODE = {"abt_buy": "abt", "amazon_google": "amgo", "dblp_acm": "dbac", "dblp_scholar": "dbgo",
                 "itunes_amazon": "itam", "walmart_amazon": "waam", "beer": "beer"}
DBLP = {"dblp_acm", "dblp_scholar"}
TITLE_NOTE = "target test titles found in the external corpora: "


def official_test(dataset):
    with zipfile.ZipFile(BASE / "anymatch-transfer-data/downloads" / ARCHIVES[dataset]) as archive:
        (member,) = [n for n in archive.namelist() if n.split("/")[-1] == "test.csv" and "__MACOSX" not in n]
        return pd.read_csv(io.BytesIO(archive.read(member)), dtype=str)[["ltable_id", "rtable_id", "label"]]


def anymatch_contract(folder, result):
    """Confirm one AnyMatch cell used the fixed cutoff and an external-validation checkpoint.

    Args:
        folder: Directory of one official-baselines/anymatch/<dataset> cell.
        result: Parsed result.json for that cell.

    Returns:
        A pair (fields, failures). fields records the threshold, checkpoint, title overlap,
        DBLP exclusion, gradient checkpointing and GPU cap. failures names every broken part.
    """
    failures = []
    fixed = (result["threshold"] == 0.5 and result["seeds"] == [131]
             and "fixed 0.5" in result["threshold_source"] and "no target labels" in result["threshold_source"])
    if not fixed:
        failures.append("fixed_threshold_0_5")
    curve = [json.loads(line) for line in (folder / "source_fit" / "learning_curve.jsonl").read_text().splitlines() if line.strip()]
    best = max(row["validation_f1"] for row in curve)
    winners = [row["epoch"] for row in curve if abs(row["validation_f1"] - best) < 1e-12]
    valid = pd.read_csv(folder / "source_fit" / "best_external_validation_predictions.csv")
    scored = f1_score(valid.label.astype(int), (valid.score >= 0.5).astype(int), zero_division=0)
    checkpoint = (winners == [result["best_epoch"]]
                  and abs(best - result["best_external_validation_f1"]) < 1e-12
                  and abs(scored - result["best_external_validation_f1"]) < 1e-12)
    if not checkpoint:
        failures.append("external_validation_checkpoint")
    overlap = json.loads(next(note for note in result["notes"] if note.startswith(TITLE_NOTE))[len(TITLE_NOTE):])
    if overlap["external_train"] != 0 or overlap["external_valid"] != 0:
        failures.append("test_title_overlap")
    protocol = json.loads((folder / "source_fit" / "protocol.json").read_text())
    left = protocol["leave_one_out_check"]["left_out_domains"]
    leak = None
    if result["dataset"] in DBLP:
        rule = result["notes"][0]
        found = re.search(r"leaks (\d+) ", rule)
        leak = int(found.group(1)) if found and "both DBLP domains (dbac, dbgo) are excluded" in rule else None
        if left != ["dbac", "dbgo"] or leak is None:
            failures.append("dblp_both_excluded")
    elif left != [ANYMATCH_CODE[result["dataset"]]]:
        failures.append("leave_one_out")
    cap = result["resource_gate_adaptation"]["cap_gib"]
    checkpointing = protocol.get("gradient_checkpointing") is True
    if abs(cap - 3.3) > 1e-9:
        failures.append("gpu_cap")
    if not checkpointing:
        failures.append("gradient_checkpointing")
    fields = {"fixed_threshold_0_5": fixed, "external_validation_checkpoint": checkpoint,
              "test_title_overlap": int(overlap["external_train"]),
              "test_title_overlap_valid": int(overlap["external_valid"]),
              "left_out_domains": left, "standard_loo_leak": leak,
              "gradient_checkpointing": checkpointing, "gpu_cap_gib": cap}
    return fields, failures


def main():
    summary = pd.read_csv(BASE / "experiments/completion/official_baselines.csv")
    checks, failures = [], []
    for row in summary.itertuples():
        folder = ROOT / Path(row.result).parent.relative_to("official-baselines")
        result = json.loads((folder / "result.json").read_text())
        frame = pd.read_csv(folder / "test_predictions.csv", dtype={"ltable_id": str, "rtable_id": str})
        test = official_test(row.dataset)
        same = (len(frame) == len(test) and frame.ltable_id.eq(test.ltable_id).all()
                and frame.rtable_id.eq(test.rtable_id).all() and frame.label.astype(int).eq(test.label.astype(int)).all())
        rule = (frame.score > result["threshold"]) if row.method == "ditto" else (frame.score >= result["threshold"])
        y, p = frame.label.astype(int), frame.prediction.astype(int)
        metrics = {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
                   "f1": f1_score(y, p, zero_division=0)}
        check = {"method": row.method, "dataset": row.dataset, "rows": len(frame), "positives": int(y.sum()),
                 "official_rows": len(test), "official_positives": int(test.label.astype(int).sum()),
                 "identical_to_archive_test_csv": bool(same), "decision_rule_holds": bool(rule.astype(int).eq(p).all()),
                 "sha256_matches_result": hashlib.sha256((folder / "test_predictions.csv").read_bytes()).hexdigest()
                 == result["test_predictions_sha256"], **metrics,
                 "f1_matches_result": abs(metrics["f1"] - result["test"]["f1"]) < 1e-12,
                 "f1_matches_csv": abs(metrics["f1"] - row.f1) < 5e-7,
                 "excluded": result["excluded_input_attributes"]}
        bad = [k for k in ["identical_to_archive_test_csv", "decision_rule_holds", "sha256_matches_result",
                           "f1_matches_result", "f1_matches_csv"] if not check[k]]
        if "id" not in check["excluded"]:
            bad.append("id not excluded")
        if row.method == "anymatch":
            fields, extra = anymatch_contract(folder, result)
            check.update(fields)
            bad += extra
        check["failures"] = bad
        failures += [f"{row.method}/{row.dataset}: {b}" for b in bad]
        checks.append(check)
        print(f"{row.method:15s} {row.dataset:15s} n={len(frame):5d} pos={int(y.sum()):5d} F1={metrics['f1']:.4f} "
              f"{'OK' if not bad else 'FAIL ' + ','.join(bad)}")
    (ROOT / "verification.json").write_text(json.dumps({"checks": checks, "failures": failures}, indent=2) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
