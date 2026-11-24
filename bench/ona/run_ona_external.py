"""External DeepMatcher benchmarks on their official splits: ONA and the additive NARS/LLR control.

For each Table 5 dataset (run_matchgpt_external.DATASETS; Fodors-Zagats without its
label-revealing ``class``, exactly the run_fodors_official split and exclusions, through
run_hiergat_external.read_splits) the executed NARS/LLR evaluator
(vendor/bench/scripts/run_nars_full_eval.py evaluate_nars_on_dataset, via run_fodors_official's
OfficialSplit harness and the pilot's dataset binding run_ona_pilot.official_external) runs twice:

1. NARS/LLR as is: counts from the official train split, threshold maximising validation F1
   (select_threshold_by_f1), the official test split scored once. This is the like-for-like
   additive control (the archived nars_full_eval.csv rows used unretained sampled pairs).
2. ONA: the same evaluator, the same fitted counts, but the pair score is ONA's answer
   expectation for the pair's episode (ona_cache: implications from the train counts,
   constant pair term, one reset episode per distinct input). The model class only replaces
   score_pair by a lookup of ONA's printed answer; an unanswered pair raises, so no cell is
   computed from a filled score. Threshold from the validation split, test scored once.

Outputs: experiments/ona/full/external/<dataset>/ and experiments/ona/full/external/results.csv.
Usage: python run_ona_external.py [DATASET ...]   (default: all eight)
"""
from __future__ import annotations

import argparse
import json
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
from run_ona_full import OUT, dump, knowledge_rows  # noqa: E402

EXT = OUT / "external"
SCRIPT = BASE / "vendor/bench/scripts/run_nars_full_eval.py"
FIELDS = ["precision", "recall", "f1", "accuracy", "threshold", "auc", "ari_cc", "ari_ag", "p_at_k", "r_at_k",
          "n_test_pairs", "n_positives"]


class Unanswered(RuntimeError):
    pass


def evaluate(dataset: str, rfo, model_class):
    """One run of the canonical evaluator with ``model_class`` as its NarsModel."""
    namespace = rfo.bench_namespace()
    created = []

    class Recorded(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
    namespace["NarsModel"] = Recorded
    rfo.canonical(SCRIPT, ["evaluate_nars_on_dataset", "select_threshold_by_f1", "build_clusters_from_pairs",
                           "compute_ari_from_pairs"], namespace)
    thresholds, pairwise = [], []
    rfo.recorder(namespace, "select_threshold_by_f1", thresholds)
    rfo.recorder(namespace, "compute_pairwise_metrics", pairwise)
    result = namespace["evaluate_nars_on_dataset"](dataset.upper().replace("_", "-"), rfo.OfficialSplit(dataset))
    split = rfo.OfficialSplit(dataset).load()
    rfo.validation_only(thresholds, pairwise, split)
    if len(created) != 1:
        raise RuntimeError("Evaluator created more than one model")
    model = created[0]
    if model.preprocess_mode != "deepmatcher" or model.dataset_name != dataset:
        raise RuntimeError("NARS did not use its DeepMatcher judgments for " + dataset)
    return {"result": result, "model": model, "split": split, "threshold": thresholds[0]["value"],
            "valid_scores": np.asarray(thresholds[0]["args"][0], dtype=float),
            "test_scores": np.asarray(pairwise[0]["kwargs"]["y_scores"], dtype=float)}


def run_dataset(dataset: str, store: oc.EpisodeStore, workers: int) -> dict:
    sys.path.insert(0, str(BASE / "vendor/bench"))
    from bench.models.nars import NarsModel
    from run_experiments import metrics
    from run_ona_pilot import official_external
    out = EXT / dataset
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with official_external(dataset)() as rfo:
        _, _, source = rfo.official_split()   # excluded columns absent, official test size
        llr = evaluate(dataset, rfo, NarsModel)
        model, split = llr["model"], llr["split"]
        records = {**{f"L_{r['id']}": dict(r) for _, r in split.left_table.iterrows()},
                   **{f"R_{r['id']}": dict(r) for _, r in split.right_table.iterrows()}}
        vocabulary = sorted(set(model.stats.match_counts) | set(model.stats.non_match_counts))
        if any(j.startswith(("class", "id_")) for j in vocabulary):
            raise RuntimeError(f"Excluded attribute in judgments: {vocabulary}")
        knowledge = om.implications_from_counts(model.stats, "decomposition")
        frames, requests = {}, {}
        for partition, pairs, llr_scores in [("valid", split.val_pairs, llr["valid_scores"]),
                                             ("test", split.test_pairs, llr["test_scores"])]:
            judgments = [frozenset(model._preprocess(records[f"L_{a}"], records[f"R_{b}"]))
                         for a, b in pairs[["ltable_id", "rtable_id"]].itertuples(index=False)]
            recomputed = np.asarray([model.stats.compute_score(sorted(j)) for j in judgments])
            if len(recomputed) != len(llr_scores) or np.max(np.abs(recomputed - llr_scores)) > 1e-12:
                raise RuntimeError("Judgments differ from the ones the LLR evaluator scored")
            digests = [store.request(j, knowledge) for j in judgments]
            for j, d in zip(judgments, digests):
                requests[j] = d
            frame = pairs[["ltable_id", "rtable_id", "label"]].copy().reset_index(drop=True)
            frame["rtable_id"] = frame.rtable_id.astype(int) - rfo.RIGHT_OFFSET
            frame["judgments"] = [" ".join(sorted(j)) for j in judgments]
            frame["input_sha256"] = digests
            frame["llr_score"] = llr_scores
            frames[partition] = frame
        ran = store.run(workers=workers)
        answers = {j: store.get(d) for j, d in requests.items()}

        class OnaScored(NarsModel):
            """NarsModel whose pair score is ONA's printed answer for the pair's episode."""
            checked = False

            def score_pair(self, rec1, rec2):
                if not OnaScored.checked:
                    if knowledge_rows(om.implications_from_counts(self.stats, "decomposition")) != knowledge_rows(knowledge):
                        raise RuntimeError("ONA run fitted other counts than the LLR run")
                    OnaScored.checked = True
                row = answers[frozenset(self._preprocess(rec1, rec2))]
                if not row["answered"]:
                    raise Unanswered(f"ONA gave no answer for judgments {row['judgments']}")
                return row["expectation"]
        try:
            ona = evaluate(dataset, rfo, OnaScored)
        except Unanswered as error:
            ona = {"error": str(error)}
    for partition, frame in frames.items():
        rows = [store.get(d) for d in frame.input_sha256]
        frame["episode_id"] = [r["episode_id"] for r in rows]
        frame["answered"] = [r["answered"] for r in rows]
        frame["ona_score"] = [r["expectation"] for r in rows]
        frame["anomaly"] = [r["anomaly"] for r in rows]
        frame.to_csv(out / f"{partition}_scores.csv.gz", index=False)
    result = {"dataset": dataset, "runtime": om.verify_runtime(), "design": oc.DESIGN,
              "split": {"pairs": {p: len(getattr(split, f"{p}_pairs")) for p in ["train", "val", "test"]},
                        "test_positive": int(split.test_pairs.label.sum()), "input_columns": list(split.left_table.columns),
                        "official_split": source},
              "knowledge": knowledge_rows(knowledge), "judgment_vocabulary": vocabulary,
              "llr_evaluator": {k: llr["result"][k] for k in FIELDS},
              "llr": metrics(frames["test"].label.to_numpy(), frames["test"].llr_score.to_numpy(), llr["threshold"]),
              "ona_seconds": ran["seconds"], "new_episodes": ran["new_episodes"], "wall_seconds": time.time() - started}
    episodes = pd.DataFrame([store.get(d) for d in set(requests.values())])
    result["episodes"] = {"distinct": len(episodes), "anomalies": episodes.anomaly.value_counts().to_dict(),
                          "max_judgments": int(episodes.n_judgments.max()),
                          "max_abs_deviation_expectation": float(episodes.dev_expectation.max())}
    result["coverage"] = {p: {"pairs": len(f), "answered": int(f.answered.sum())} for p, f in frames.items()}
    if "error" in ona:
        result["ona"] = f"not computed: {ona['error']}"
    else:
        for partition, key in [("valid", "valid_scores"), ("test", "test_scores")]:
            if not np.array_equal(ona[key], frames[partition].ona_score.to_numpy()):
                raise RuntimeError("Evaluator ONA scores differ from the per-pair ONA answers")
        result["ona_evaluator"] = {k: ona["result"][k] for k in FIELDS}
        result["ona"] = metrics(frames["test"].label.to_numpy(), frames["test"].ona_score.to_numpy(), ona["threshold"])
    if dataset == "fodors_zagats":
        archived = json.loads((BASE / "experiments/completion/fodors-official/nars/result.json").read_text())
        saved = pd.read_csv(BASE / "experiments/completion/fodors-official/nars/test_predictions.csv")
        same = (np.array_equal(saved.ltable_id.astype(int), frames["test"].ltable_id.astype(int)) and
                np.array_equal(saved.rtable_id.astype(int), frames["test"].rtable_id) and
                float(np.max(np.abs(saved.score - frames["test"].llr_score))) <= 1e-12)
        result["llr_reproduces_completion_fodors_official"] = {
            "evaluator_fields_equal": {k: bool(archived["evaluator_result"][k] == llr["result"][k]) for k in FIELDS},
            "test_scores_equal": bool(same)}
        if not same or not all(result["llr_reproduces_completion_fodors_official"]["evaluator_fields_equal"].values()):
            raise RuntimeError("LLR rerun does not reproduce the completed Fodors-Zagats official run")
    dump(out / "result.json", result)
    return result


def main(datasets, workers):
    from run_matchgpt_external import DATASETS
    store = oc.EpisodeStore(OUT / "episodes/external")
    rows = []
    for dataset in datasets or list(DATASETS):
        if dataset not in DATASETS:
            raise SystemExit(f"Unknown dataset {dataset}")
        r = run_dataset(dataset, store, workers)
        for engine, key in [("published_nars", "llr_evaluator"), ("ona", "ona_evaluator")]:
            if key in r:
                standard = r["llr" if engine == "published_nars" else "ona"]
                rows.append({"dataset": dataset, "model": engine, **r[key],
                             "auc_tie_averaged": standard["auc"],   # sklearn; the evaluator's auc breaks ties by input order
                             "average_precision": standard["average_precision"], "tp": standard["tp"],
                             "fp": standard["fp"], "fn": standard["fn"], "tn": standard["tn"],
                             "split": "official DeepMatcher train/valid/test" + (" without class" if dataset == "fodors_zagats" else ""),
                             "episodes": r["episodes"]["distinct"] if engine == "ona" else None,
                             "anomalous_episodes": sum(v for k, v in r["episodes"]["anomalies"].items() if k) if engine == "ona" else None})
            else:
                rows.append({"dataset": dataset, "model": engine, "note": r["ona"]})
        print(json.dumps({k: r[k] for k in ["dataset", "llr_evaluator", "coverage", "episodes"]} |
                         {"ona_evaluator": r.get("ona_evaluator", r["ona"])}), flush=True)
    store.export()
    frame = pd.DataFrame(rows)
    path = EXT / "results.csv"
    if path.exists() and datasets:
        old = pd.read_csv(path)
        frame = pd.concat([old[~old.dataset.isin(frame.dataset)], frame], ignore_index=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.datasets, args.workers)
