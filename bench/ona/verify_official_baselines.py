"""Independent check of experiments/completion/official-baselines (no code shared with the drivers).

For every <method>/<dataset>/test_predictions.csv: the rows must be the official DeepMatcher test.csv
read straight from the archive (ids, labels, order); P/R/F1 are recomputed with scikit-learn from the
saved decisions and compared with result.json and official_baselines.csv; the decision must equal
score >= threshold (Ditto: score > threshold, its evaluate() rule). Writes verification.json.
"""
import hashlib
import io
import json
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


def official_test(dataset):
    with zipfile.ZipFile(BASE / "anymatch-transfer-data/downloads" / ARCHIVES[dataset]) as archive:
        (member,) = [n for n in archive.namelist() if n.split("/")[-1] == "test.csv" and "__MACOSX" not in n]
        return pd.read_csv(io.BytesIO(archive.read(member)), dtype=str)[["ltable_id", "rtable_id", "label"]]


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
        check["failures"] = bad
        failures += [f"{row.method}/{row.dataset}: {b}" for b in bad]
        checks.append(check)
        print(f"{row.method:15s} {row.dataset:15s} n={len(frame):5d} pos={int(y.sum()):5d} F1={metrics['f1']:.4f} "
              f"{'OK' if not bad else 'FAIL ' + ','.join(bad)}")
    (ROOT / "verification.json").write_text(json.dumps({"checks": checks, "failures": failures}, indent=2) + "\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
