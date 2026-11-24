"""Full replacement run: every NARS/LLR protocol of the papers scored by ONA (ona_matcher.py).

Results only; outputs under experiments/ona/full/. Subcommands:

* ``iceid-inputs`` (``python source_alias.py run_ona_full.py iceid-inputs``): observed records
  (experiment_data.load_inputs + fitted_models.observed_records, the publication path) and
  reference persons of every endpoint of every saved ICE-ID pair file, written once.
* ``iceid``: per seed, re-fit every NARS/LLR count model on its own training pairs through
  fitted_models.fit_model (published_nars on the controlled training pairs, published_nars on the
  candidate-matched training pairs, ablate_<category> on the controlled training pairs) and check
  the counts equal the saved models'; judgments of every saved pair (controlled
  train/selection/calibration/test, 136-pair audit, candidate graph selection/calibration/test,
  ablation selection/test, locked controlled and candidate cases); LLR scores recomputed and
  checked against every saved prediction file; ONA answers through the exact-input episode
  cache (ona_cache.py); per-pair score files.
* ``fresh-check``: >= 2,000 sampled pairs across all protocols re-run as per-pair episodes.
* ``saturation``: concept-memory statistics of sampled distinct episodes.
* ``explain``: ONA explanations of the two seed-131 pairs of the paper's explanation figures.
* metrics: ``python run_ona_metrics.py``; external benchmarks: ``python run_ona_external.py``.
"""
from __future__ import annotations

import argparse
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

OUT = BASE / "experiments/ona/full"
ICE = OUT / "iceid"
SEEDS = [131, 132, 133]
PUB = BASE / "experiments/publication-quarantined-extended"
CAND = BASE / "experiments/candidate-quarantined-extended"
LOCKED = BASE / "plan-completion/locked-assessment-quarantined-extended"
PARTS = ["selection", "calibration", "test"]
CATEGORIES = ["name", "geography", "birthyear", "sex", "wave"]   # fitted_models.ABLATION_FIELDS order
FORM = "decomposition"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n")


def sha256(path) -> str:
    return om.sha256_file(Path(path))


def pair_files() -> dict:
    """Every saved ICE-ID pair file a NARS/LLR cell was computed on (name -> path)."""
    files = {}
    for seed in SEEDS:
        sd, vd = PUB / f"seed_{seed}", PUB / f"seed_{seed}/corrected_training_labels"
        files[f"{seed}/controlled/train"] = vd / "train_pairs.csv"
        for p in PARTS:
            files[f"{seed}/controlled/{p}"] = sd / f"{p}_pairs.csv"
            files[f"{seed}/graph/{p}"] = sd / f"{p}_graph_pairs.csv.gz"
            files[f"{seed}/graph_nodes/{p}"] = sd / f"{p}_graph_nodes.csv"
        files[f"{seed}/controlled/audit"] = vd / "audit_predictions.csv"
        files[f"{seed}/candidate/train"] = CAND / f"seed_{seed}/train_pairs.csv"
    files["locked/controlled"] = LOCKED / "controlled_pairs.csv"
    files["locked/candidate"] = LOCKED / "candidate_pairs.csv.gz"
    files["locked/nodes"] = LOCKED / "candidate_nodes.csv"
    return files


# ------------------------------------------------------------------ inputs (people table once)
def iceid_inputs():
    from prepare_release import configured_path
    from experiment_data import load_inputs
    from fitted_models import observed_records
    from resource_limits import check_memory
    check_memory(6)
    files = pair_files()
    needed = set()
    for name, path in files.items():
        columns = ["id"] if "nodes" in name else ["id1", "id2"]
        frame = pd.read_csv(path, usecols=columns)
        needed |= set(frame.to_numpy().ravel().astype(int))
    needed = sorted(needed)
    release = configured_path("selected_release")
    people, _, _ = load_inputs(release)
    records = observed_records(people.loc[needed])
    persons = people.loc[needed, ["person"]].astype("int64")
    del people
    target = ICE / "inputs"
    target.mkdir(parents=True, exist_ok=True)
    with gzip.open(target / "records.json.gz", "wt") as stream:
        json.dump({str(k): v for k, v in records.items()}, stream, sort_keys=True)
    persons.rename_axis("id").to_csv(target / "persons.csv.gz")
    dump(target / "manifest.json", {
        "records": len(records), "release": str(release),
        "release_people_sha256": sha256(release / "raw_data/people.csv"),
        "pair_files_sha256": {k: sha256(v) for k, v in files.items()},
        "builder": "experiment_data.load_inputs + fitted_models.observed_records (publication path); "
                   "persons = load_inputs people.person (the component_bootstrap identity column)"})
    print(f"cached {len(records)} records")


def load_records() -> dict:
    with gzip.open(ICE / "inputs/records.json.gz", "rt") as stream:
        return {int(k): v for k, v in json.load(stream).items()}


def load_persons() -> pd.DataFrame:
    return pd.read_csv(ICE / "inputs/persons.csv.gz", index_col="id")


def check_inputs():
    manifest = json.loads((ICE / "inputs/manifest.json").read_text())
    for name, path in pair_files().items():
        if sha256(path) != manifest["pair_files_sha256"][name]:
            raise RuntimeError(f"Saved pair file changed since the record cache: {name}")


# ------------------------------------------------------------------ knowledge
def knowledge_rows(knowledge: dict) -> list:
    return [{"judgment": i.judgment, "w_plus": i.positive, "w": i.total, "frequency": i.frequency,
             "confidence": i.confidence, "negated": i.negated, "form": i.form, "narsese": i.narsese}
            for i in knowledge.values()]


def knowledge_from_rows(rows: list) -> dict:
    return {r["judgment"]: om.Implication(r["judgment"], int(r["w_plus"]), int(r["w"]), float(r["frequency"]),
                                          float(r["confidence"]), bool(r["negated"]), r["form"]) for r in rows}


def same_counts(a, b) -> bool:
    keys = set(a.match_counts) | set(a.non_match_counts) | set(b.match_counts) | set(b.non_match_counts)
    return (a.total_matches == b.total_matches and a.total_non_matches == b.total_non_matches and
            all(a.match_counts.get(k, 0) == b.match_counts.get(k, 0) and
                a.non_match_counts.get(k, 0) == b.non_match_counts.get(k, 0) for k in keys))


def fit_models(seed: int, records: dict) -> tuple[dict, dict]:
    """Re-fit every count model of this seed on its own training pairs (fitted_models.fit_model)."""
    import joblib
    from fitted_models import fit_model
    fits, report = {}, {}
    plan = [("pub", "published_nars", PUB / f"seed_{seed}/corrected_training_labels"),
            ("cand", "published_nars", CAND / f"seed_{seed}")]
    plan += [(f"abl_{c}", f"ablate_{c}", PUB / f"seed_{seed}/corrected_training_labels") for c in CATEGORIES]
    for kb, name, source in plan:
        train = pd.read_csv(source / "train_pairs.csv")
        target = ICE / f"models/seed_{seed}/{kb}"
        start = time.time()
        model = fit_model(name, train, None, records, target, seed)
        seconds = time.time() - start
        saved = joblib.load(source / f"{name}.joblib")
        if not same_counts(model.stats, saved.stats) or model.exclude_judgments != saved.exclude_judgments:
            raise RuntimeError(f"Re-fitted counts differ from the saved {seed}/{kb} model")
        if model.preprocess_mode != "iceid":
            raise RuntimeError("ICE-ID model is not in iceid judgment mode")
        fits[kb] = model
        report[kb] = {"model": name, "training_pairs": str(source / "train_pairs.csv"), "n_training_pairs": len(train),
                      "training_positive": int(train.label.sum()), "fit_seconds": seconds,
                      "saved_model": str(source / f"{name}.joblib"), "counts_equal_saved_model": True,
                      "exclude_judgments": model.exclude_judgments}
    return fits, report


# ------------------------------------------------------------------ ICE-ID scoring
class Sets:
    """Registry of distinct judgment sets."""

    def __init__(self):
        self.ids, self.sets = {}, []

    def add(self, judgments) -> int:
        key = frozenset(judgments)
        if key not in self.ids:
            self.ids[key] = len(self.sets)
            self.sets.append(key)
        return self.ids[key]


def judgments_for(model, records, frame, cache: dict, sets: Sets, mode: str) -> np.ndarray:
    out = np.empty(len(frame), dtype=np.int64)
    for i, (a, b) in enumerate(frame[["id1", "id2"]].itertuples(index=False, name=None)):
        key = (mode, int(a), int(b))
        if key not in cache:
            cache[key] = sets.add(model._preprocess(records[int(a)], records[int(b)]))
        out[i] = cache[key]
    return out


def protocol_frames(seed: int) -> list:
    """(file stem, partition, kb, judgment mode, frame) for every scored population of a seed."""
    sd, vd = PUB / f"seed_{seed}", PUB / f"seed_{seed}/corrected_training_labels"
    out = [("controlled", "train", "pub", "full", pd.read_csv(vd / "train_pairs.csv"))]
    out += [("controlled", p, "pub", "full", pd.read_csv(sd / f"{p}_pairs.csv")) for p in PARTS]
    audit = pd.read_csv(vd / "audit_predictions.csv", usecols=["id1", "id2", "label", "wave"])
    out.append(("controlled", "audit", "pub", "full", audit))
    for p in PARTS:
        graph = pd.read_csv(sd / f"{p}_graph_pairs.csv.gz")
        out.append(("publication_graph", p, "pub", "full", graph))
        out.append(("candidate_graph", p, "cand", "full", graph))
    for c in CATEGORIES:
        for p in ["selection", "test"]:
            out.append((f"ablate_{c}", p, f"abl_{c}", f"abl_{c}", pd.read_csv(sd / f"{p}_pairs.csv")))
    out.append(("locked_controlled", "locked", "pub", "full", pd.read_csv(LOCKED / "controlled_pairs.csv")))
    out.append(("locked_candidate", "locked", "cand", "full", pd.read_csv(LOCKED / "candidate_pairs.csv.gz")))
    return out


def saved_llr(seed: int) -> list:
    """(file stem, partition, saved frame, score column) of every saved LLR prediction file."""
    vd = PUB / f"seed_{seed}/corrected_training_labels"
    out = [("controlled", "test", pd.read_csv(vd / "test_predictions.csv", float_precision="round_trip"), "published_nars"),
           ("controlled", "audit", pd.read_csv(vd / "audit_predictions.csv", float_precision="round_trip"), "published_nars"),
           ("publication_graph", "test", pd.read_csv(vd / "graph_predictions.csv.gz", float_precision="round_trip"), "published_nars"),
           ("candidate_graph", "test", pd.read_csv(CAND / f"seed_{seed}/graph_predictions.csv.gz", float_precision="round_trip"), "published_nars")]
    test = out[0][2]
    out += [(f"ablate_{c}", "test", test, f"ablate_{c}") for c in CATEGORIES]
    for pop in ["controlled", "candidate"]:
        out.append((f"locked_{pop}", "locked", pd.read_csv(LOCKED / f"seed_{seed}/{pop}_published_nars_predictions.csv.gz",
                                                           float_precision="round_trip"), "score"))
    return out


def iceid(workers: int):
    check_inputs()
    runtime = om.verify_runtime()
    records = load_records()
    sets, cache = Sets(), {}
    store = oc.EpisodeStore(OUT / "episodes/iceid")
    plans, fit_report, knowledge_all = [], {}, {}
    start = time.time()
    for seed in SEEDS:
        fits, fit_report[seed] = fit_models(seed, records)
        knowledge = {kb: om.implications_from_counts(m.stats, FORM) for kb, m in fits.items()}
        knowledge_all[seed] = {kb: knowledge_rows(k) for kb, k in knowledge.items()}
        for stem, partition, kb, mode, frame in protocol_frames(seed):
            model = fits[kb] if mode != "full" else fits["pub"]
            ids = judgments_for(model, records, frame, cache, sets, mode)
            plans.append((seed, stem, partition, kb, frame, ids))
        if fits["cand"].exclude_judgments or fits["pub"].exclude_judgments:
            raise RuntimeError("Full-vocabulary models must not exclude judgments")
        print(f"seed {seed}: judgments of {sum(len(p[4]) for p in plans if p[0] == seed)} pairs, "
              f"{len(sets.sets)} distinct sets so far", flush=True)
    dump(ICE / "knowledge.json", knowledge_all)
    dump(ICE / "fits.json", fit_report)
    llr, digest = {}, {}
    for seed, stem, partition, kb, frame, ids in plans:
        digest.setdefault((seed, kb), {})
        # The knowledge as archived (knowledge.json), so the archive is exactly what ONA was given.
        k = knowledge_from_rows(knowledge_all[seed][kb])
        for s in np.unique(ids):
            if s not in digest[(seed, kb)]:
                digest[(seed, kb)][s] = store.request(sets.sets[s], k)
    # LLR control from the saved joblibs themselves (nars_scores sums in sorted order), checked
    # against every saved LLR prediction file before any ONA episode runs.
    import joblib
    saved_models = {}
    for seed in SEEDS:
        vd = PUB / f"seed_{seed}/corrected_training_labels"
        saved_models[(seed, "pub")] = joblib.load(vd / "published_nars.joblib")
        saved_models[(seed, "cand")] = joblib.load(CAND / f"seed_{seed}/published_nars.joblib")
        for c in CATEGORIES:
            saved_models[(seed, f"abl_{c}")] = joblib.load(vd / f"ablate_{c}.joblib")
    llr_scores = {}
    for seed, stem, partition, kb, frame, ids in plans:
        model, table = saved_models[(seed, kb)], llr.setdefault((seed, kb), {})
        for s in np.unique(ids):
            if s not in table:
                table[s] = float(model.stats.compute_score(sorted(sets.sets[s])))
        llr_scores[(seed, stem, partition)] = (frame, np.asarray([table[s] for s in ids]))
    checks = []
    for seed in SEEDS:
        for stem, partition, saved, column in saved_llr(seed):
            frame, mine = llr_scores[(seed, stem, partition)]
            same_cases = (len(frame) == len(saved) and np.array_equal(frame.id1, saved.id1) and
                          np.array_equal(frame.id2, saved.id2) and np.array_equal(frame.label, saved.label))
            diff = float(np.max(np.abs(mine - saved[column].to_numpy()))) if same_cases else None
            checks.append({"seed": seed, "file": stem, "partition": partition, "column": column, "pairs": len(saved),
                           "same_cases_in_order": same_cases, "max_abs_llr_difference": diff})
            if not same_cases or diff > 1e-12:
                raise RuntimeError(f"Recomputed LLR differs from saved predictions: {checks[-1]}")
    print(f"LLR recomputation equals all {len(checks)} saved prediction files; "
          f"distinct episodes requested: {len(store.records) + len(store.pending)} "
          f"({len(store.pending)} new); judgment sets {len(sets.sets)}", flush=True)
    ran = store.run(workers=workers)
    print(f"ONA ran {ran['new_episodes']} episodes in {ran['seconds']:.0f} s", flush=True)
    episodes = store.export()
    by_digest = store.records
    written = {}
    for seed, stem, partition, kb, frame, ids in plans:
        out = frame[[c for c in ["id1", "id2", "candidate_rank", "label", "wave"] if c in frame.columns]].copy()
        out.insert(0, "partition", partition)
        out.insert(0, "seed", seed)
        out["kb"] = kb
        out["judgment_set"] = ids
        dig = [digest[(seed, kb)][s] for s in ids]
        out["episode_id"] = [by_digest[d]["episode_id"] for d in dig]
        out["answered"] = [by_digest[d]["answered"] for d in dig]
        out["ona_score"] = [by_digest[d]["expectation"] for d in dig]
        out["anomaly"] = [by_digest[d]["anomaly"] for d in dig]
        out["llr_score"] = llr_scores[(seed, stem, partition)][1]
        written.setdefault((seed, stem), []).append(out)
    for (seed, stem), parts in written.items():
        frame = pd.concat(parts, ignore_index=True)
        path = ICE / f"pairs/seed_{seed}/{stem}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        back = pd.read_csv(path, float_precision="round_trip")
        if not (np.array_equal(back.llr_score, frame.llr_score) and
                np.array_equal(back.ona_score.to_numpy(), frame.ona_score.to_numpy(), equal_nan=True)):
            raise RuntimeError(f"Per-pair scores changed on CSV readback: {path}")
    sets_frame = pd.DataFrame({"judgment_set": range(len(sets.sets)),
                               "judgments": [" ".join(sorted(x)) for x in sets.sets]})
    sets_frame.to_csv(ICE / "judgment_sets.csv.gz", index=False)
    pairs_total = sum(len(f) for _, _, _, _, f, _ in plans)
    anomalies = episodes.anomaly.fillna("").value_counts().to_dict()
    dump(ICE / "scoring.json", {
        "runtime": runtime, "design": oc.DESIGN, "cache": {"pair_term": oc.TERM, "episodes": len(episodes),
                                                            "judgment_sets": len(sets.sets), "pairs_scored": pairs_total},
        "episode_anomalies": anomalies, "llr_saved_prediction_checks": checks,
        "ona_seconds_this_run": ran["seconds"], "wall_seconds": time.time() - start})
    print(json.dumps({"pairs": pairs_total, "episodes": len(episodes), "anomalies": anomalies,
                      "llr_checks": len(checks)}, indent=1))


# ------------------------------------------------------------------ cache equivalence
ICE_STEMS = {"controlled": 100, "publication_graph": 100, "candidate_graph": 100,
             **{f"ablate_{c}": 20 for c in CATEGORIES}, "locked_controlled": 50, "locked_candidate": 50}
EXTERNAL_PER_DATASET = 75


def iceid_knowledge(seed, kb):
    return knowledge_from_rows(json.loads((ICE / "knowledge.json").read_text())[str(seed)][kb])


def fresh_check(distinct_extra: int = 300, own_process: int = 64):
    """>= 2,000 random pairs across all protocols: cached answer versus a fresh per-pair episode."""
    runtime = om.verify_runtime()
    rng = np.random.default_rng(20261124)
    out = OUT / "cache_check"
    report, all_rows = {"runtime": runtime, "sampling": {}}, []
    knowledge_json = json.loads((ICE / "knowledge.json").read_text())
    for domain in ["iceid", "external"]:
        store = oc.EpisodeStore(OUT / f"episodes/{domain}")
        by_id = store.records_by_id
        samples, covered = [], set()
        if domain == "iceid":
            for seed in SEEDS:
                for stem, quota in ICE_STEMS.items():
                    frame = pd.read_csv(ICE / f"pairs/seed_{seed}/{stem}.csv.gz",
                                        usecols=["seed", "partition", "kb", "id1", "id2", "episode_id"])
                    for i in rng.choice(len(frame), quota, replace=False):
                        r = frame.iloc[int(i)]
                        k = knowledge_from_rows(knowledge_json[str(seed)][r.kb])
                        samples.append((f"iceid/{seed}/{stem}/{r.partition}/{r.id1}-{r.id2}", int(r.episode_id), k))
            kb_of = {}
            for seed in SEEDS:
                for stem in ICE_STEMS:
                    f = pd.read_csv(ICE / f"pairs/seed_{seed}/{stem}.csv.gz", usecols=["kb", "episode_id"]).drop_duplicates()
                    for kb, e in f.itertuples(index=False):
                        kb_of.setdefault(int(e), (seed, kb, stem))
        else:
            kb_of = {}
            for result_path in sorted(EXT_ROOT().glob("*/result.json")):
                dataset = result_path.parent.name
                k = knowledge_from_rows(json.loads(result_path.read_text())["knowledge"])
                frame = pd.concat([pd.read_csv(result_path.parent / f"{p}_scores.csv.gz").assign(partition=p)
                                   for p in ["valid", "test"]], ignore_index=True)
                for i in rng.choice(len(frame), EXTERNAL_PER_DATASET, replace=False):
                    r = frame.iloc[int(i)]
                    samples.append((f"external/{dataset}/{r.partition}/{r.ltable_id}-{r.rtable_id}", int(r.episode_id), k))
                for e in frame.episode_id.unique():
                    kb_of.setdefault(int(e), (dataset, "external", dataset))
        covered = {e for _, e, _ in samples}
        # Additionally one pair per randomly drawn distinct episode not hit by the random pairs.
        rest = sorted(set(kb_of) - covered)
        extra = rng.choice(rest, min(distinct_extra if domain == "iceid" else distinct_extra // 3, len(rest)), replace=False)
        for e in extra:
            seed, kb, stem = kb_of[int(e)]
            k = iceid_knowledge(seed, kb) if domain == "iceid" else knowledge_from_rows(
                json.loads((EXT_ROOT() / seed / "result.json").read_text())["knowledge"])
            samples.append((f"{domain}/{seed}/{stem}/distinct-episode", int(e), k))
        checked = []
        for label, e, k in samples:
            row = by_id[e]
            judgments = row["judgments"].split()
            if oc.sha(oc.canonical(judgments, k)) != row["input_sha256"]:
                raise RuntimeError(f"Pair map inconsistent for {label}: episode {e} is not this pair's input")
            checked.append((label, judgments, k))
        rows, timing = oc.fresh_check(checked, store, out / domain, own_process=own_process)
        for r in rows:
            r["domain"] = domain
        all_rows += rows
        frame = pd.DataFrame(rows)
        report["sampling"][domain] = {
            "pairs": len(rows), "random_pairs": len(rows) - len(extra), "distinct_episode_extra": int(len(extra)),
            "distinct_episodes_covered": int(frame.episode_id.nunique()), "episodes_in_store": len(by_id),
            "segment_identical": int(frame.segment_identical.sum()),
            "answer_line_identical": int(frame.answer_line_identical.sum()),
            "own_process_checked": int(frame.own_process_segment_identical.notna().sum()) if "own_process_segment_identical" in frame else 0,
            "own_process_identical": int(frame.own_process_segment_identical.fillna(False).sum()) if "own_process_segment_identical" in frame else 0,
            **timing}
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out / "fresh_vs_cached.csv.gz", index=False)
    report["total_pairs"] = len(frame)
    report["all_identical"] = bool(frame.segment_identical.all() and frame.answer_line_identical.all() and
                                   frame.own_process_segment_identical.dropna().astype(bool).all())
    report["comparison"] = ("complete per-episode ONA output (every line from the episode's first output to its "
                            "Answer line), per-pair term {pv<n>} and marker renamed to the constant ones; "
                            "per-pair episodes reset-separated, 100 per NAR process as in the pilot, the first "
                            f"{own_process} of each domain also alone in their own NAR process")
    dump(out / "report.json", report)
    print(json.dumps(report, indent=1, default=str))


def EXT_ROOT() -> Path:
    return OUT / "external"


def saturation(iceid_sample: int = 400):
    rng = np.random.default_rng(20261125)
    out = OUT / "saturation"
    rows = []
    for domain in ["iceid", "external"]:
        store = oc.EpisodeStore(OUT / f"episodes/{domain}")
        digests = sorted(store.records)
        if domain == "iceid":
            digests = [digests[i] for i in sorted(rng.choice(len(digests), iceid_sample, replace=False))]
        for r in oc.saturation(digests, store, out / domain):
            rows.append({"domain": domain, **r})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "saturation.csv", index=False)
    summary = frame.groupby("domain").agg(episodes=("episode_id", "size"), saturated=("saturated", "sum"),
                                          answer_identical=("answer_identical", "sum"),
                                          max_judgments=("n_judgments", "max")).reset_index().to_dict("records")
    by_size = frame.groupby(["domain", "n_judgments"]).agg(episodes=("episode_id", "size"), saturated=("saturated", "sum"),
                                                         min_concepts=("total_concepts", "min"),
                                                         anomalies=("anomaly", lambda x: int((x.fillna("") != "").sum()))
                                                         ).reset_index().to_dict("records")
    dump(out / "summary.json", {"summary": summary, "by_n_judgments": by_size, "concepts_max": 16384,
                                "note": "total concepts after the answer; *stats appended after the question"})
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ explanations
def explain():
    """ONA's answer for the two seed-131 explanation pairs as its evidence-weighted revision."""
    import joblib
    seed = SEEDS[0]
    aaai = BASE.parent / "new_paper_subs/AAAI Magazine (model paper)/figure_data/explanations.json"
    figures = json.loads(aaai.read_text())
    knowledge = iceid_knowledge(seed, "pub")
    store = oc.EpisodeStore(OUT / "episodes/iceid")
    controlled = pd.read_csv(ICE / f"pairs/seed_{seed}/controlled.csv.gz", float_precision="round_trip")
    test = controlled[controlled.partition.eq("test")]
    model = joblib.load(PUB / f"seed_{seed}/corrected_training_labels/published_nars.joblib")
    records = load_records()
    out = OUT / "explanations"
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for figure in figures:
        a, b = int(figure["id1"]), int(figure["id2"])
        row = test[test.id1.eq(a) & test.id2.eq(b)]
        if len(row) != 1:
            raise RuntimeError(f"Explanation pair {a}/{b} is not a seed-{seed} test pair")
        row = row.iloc[0]
        judgments = sorted(model._preprocess(records[a], records[b]))
        if judgments != sorted(figure["judgments"]):
            raise RuntimeError("Explanation judgments differ from the published figure")
        cached = store.records_by_id[int(row.episode_id)]
        lines = oc.episode_lines(judgments, knowledge, volume=100)
        stdout = om.run_batch(lines)
        stem = f"pair_{a}_{b}"
        (out / f"{stem}.nal").write_text("\n".join(lines) + "\n")
        (out / f"{stem}.out").write_text(stdout)
        answer = oc.parse_answer(oc.segments(stdout, om.MARK)[oc.KEY], oc.TERM)
        # Stamp ids: ONA numbers input sentences 1, 2, ... after *reset (Event_INIT).
        ids, next_id = {}, 1
        for line in lines:
            if line.startswith("<"):
                ids[next_id] = line
                next_id += 1
        derived = []
        pattern = re.compile(r"^(Derived|Revised): <\{px\} --> match>\. Priority=\S+ Stamp=\[([\d,]+)\] "
                             r"Truth: frequency=(\d+\.\d+), confidence=(\d+\.\d+)$")
        for line in stdout.splitlines():
            m = pattern.match(line)
            if m:
                derived.append({"kind": m.group(1), "stamp": [int(x) for x in m.group(2).split(",")],
                                "frequency": float(m.group(3)), "confidence": float(m.group(4)), "line": line})
        # ONA prints a judgment's conclusion as "Derived:" only while it is not yet revised into the
        # belief; later conclusions appear only as "Revised:" steps of <{px} --> match>. Each step's
        # contribution is recovered from ONA's printed chain: the stamp ids it adds name the judgment,
        # w = c/(1-c), dw = w_k - w_(k-1), and its frequency (w_k f_k - w_(k-1) f_(k-1)) / dw.
        chain = [d for d in derived if d["kind"] == "Revised"]
        first = next(d for d in derived if d["kind"] == "Derived")
        id_judgment = {}
        for j in judgments:
            if j in knowledge:
                imp_id = next(i for i, l in ids.items() if l == knowledge[j].narsese)
                judgment_id = next(i for i, l in ids.items() if l == f"<{oc.TERM} --> {j}>.")
                id_judgment[imp_id] = id_judgment[judgment_id] = j
        weight = lambda c: c / (1 - c)
        steps, previous = [], first
        first_judgments = {id_judgment[i] for i in first["stamp"]}
        if len(first_judgments) != 1:
            raise RuntimeError("First conclusion does not come from a single judgment")
        steps.append({"judgment": first_judgments.pop(), "dw": weight(first["confidence"]), "f": first["frequency"],
                      "ona_line": first["line"]})
        for step in chain:
            added = set(step["stamp"]) - set(previous["stamp"])
            named = {id_judgment[i] for i in added}
            if len(named) != 1 or len(added) != 2:
                raise RuntimeError(f"Revision step adds evidence of {named or 'no judgment'}: {step['line']}")
            w0, w1 = weight(previous["confidence"]), weight(step["confidence"])
            steps.append({"judgment": named.pop(), "dw": w1 - w0,
                          "f": (w1 * step["frequency"] - w0 * previous["frequency"]) / (w1 - w0), "ona_line": step["line"]})
            previous = step
        if previous["line"].split("Stamp=")[1] != answer["line"].split("Stamp=")[1]:
            raise RuntimeError("ONA's answer is not the last revised belief")
        by_judgment = {}
        for st in steps:
            if st["judgment"] in by_judgment:
                raise RuntimeError(f"Evidence of {st['judgment']} revised in twice")
            by_judgment[st["judgment"]] = st
        parts = []
        for j in judgments:
            if j not in knowledge:
                parts.append({"judgment": j, "implication": None, "note": "no training evidence: no implication, no conclusion"})
                continue
            if j not in by_judgment:
                raise RuntimeError(f"Evidence of {j} never reached the answer")
            imp, st = knowledge[j], by_judgment[j]
            exact_c = imp.frequency * imp.confidence * 0.9 * 0.9 * imp.frequency   # deduction, then structural
            exact_f = 1 - imp.frequency if imp.negated else imp.frequency
            parts.append({"judgment": j, "implication": imp.narsese, "w_plus": imp.positive, "w": imp.total,
                          "revision_order": steps.index(st) + 1, "ona_revision_line": st["ona_line"],
                          "frequency": exact_f, "confidence": exact_c, "evidence_weight": weight(exact_c),
                          "from_ona_chain_frequency": st["f"], "from_ona_chain_evidence_weight": st["dw"]})
        used = [p for p in parts if p.get("implication")]
        total = sum(p["evidence_weight"] for p in used)
        for p in used:
            p["share"] = p["evidence_weight"] / total
            p["frequency_contribution"] = p["share"] * p["frequency"]
            p["chain_vs_exact_abs_dw"] = abs(p["from_ona_chain_evidence_weight"] - p["evidence_weight"])
            p["chain_vs_exact_abs_f"] = abs(p["from_ona_chain_frequency"] - p["frequency"])
        f = sum(p["frequency_contribution"] for p in used)
        c = min(om.MAX_CONFIDENCE, total / (total + om.EVIDENTIAL_HORIZON))
        others = [d for d in derived if d["kind"] == "Derived" and d is not first and
                  {id_judgment.get(i) for i in d["stamp"]} - {None} and len(set(d["stamp"])) != 2]
        entry = {"id1": a, "id2": b, "reference_label": figure["reference_label"], "judgments": judgments,
                 "record1": figure["record1"], "record2": figure["record2"],
                 "ona_answer_line": answer["line"], "cached_scored_answer_line": cached["answer_line"],
                 "identical_to_scored_run": answer["line"] == cached["answer_line"],
                 "ona_frequency": answer["frequency"], "ona_confidence": answer["confidence"],
                 "ona_expectation": answer["expectation"], "ona_score_in_results": float(row.ona_score),
                 "shares_frequency": f, "shares_confidence": c, "shares_expectation": c * (f - .5) + .5,
                 "reproduces_ona_frequency": abs(f - answer["frequency"]) <= 1e-6,
                 "reproduces_ona_confidence": abs(c - answer["confidence"]) <= 1e-6,
                 "abs_difference_frequency": abs(f - answer["frequency"]),
                 "abs_difference_confidence": abs(c - answer["confidence"]),
                 "revision_steps": [d["line"] for d in derived if d["kind"] == "Revised"],
                 "max_chain_vs_exact_abs_evidence_weight": max(p["chain_vs_exact_abs_dw"] for p in used),
                 "max_chain_vs_exact_abs_frequency": max(p["chain_vs_exact_abs_f"] for p in used),
                 "other_match_derivations": [d["line"] for d in others],
                 "parts": parts, "llr_figure": {k: figure[k] for k in ["raw_score", "prior_log_odds", "sum_log_odds",
                                                                         "threshold", "judgments"]},
                 "interpretation": ("ONA revision (Truth_Revision) is an evidence-weighted average: with w_k = c_k/(1-c_k) "
                                    "of each judgment's conclusion <{pair} --> match>, f = sum(w_k f_k)/sum(w_k), "
                                    "c = sum(w_k)/(sum(w_k)+1); share_k = w_k/sum(w_k). f_k, c_k: the NAL deduction + "
                                    "structural-deduction truth of judgment k's conclusion; from_ona_chain_*: the same "
                                    "quantities recovered from ONA's own printed revision steps (6-decimal output).")}
        report.append(entry)
    dump(out / "explanations.json", report)
    rows = [{"id1": e["id1"], "id2": e["id2"], "reference_label": e["reference_label"], **{k: p.get(k) for k in [
             "judgment", "w_plus", "w", "revision_order", "frequency", "confidence", "evidence_weight", "share",
             "frequency_contribution", "from_ona_chain_frequency", "from_ona_chain_evidence_weight"]}}
            for e in report for p in e["parts"]]
    pd.DataFrame(rows).to_csv(out / "explanation_shares.csv", index=False)
    print(json.dumps([{k: e[k] for k in ["id1", "id2", "ona_answer_line", "identical_to_scored_run", "shares_frequency",
                                         "shares_confidence", "reproduces_ona_frequency", "reproduces_ona_confidence",
                                         "max_chain_vs_exact_abs_evidence_weight", "max_chain_vs_exact_abs_frequency"]}
                      for e in report], indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["iceid-inputs", "iceid", "fresh-check", "saturation", "explain"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "iceid-inputs":
        iceid_inputs()
    elif args.command == "iceid":
        iceid(args.workers)
    elif args.command == "fresh-check":
        fresh_check()
    elif args.command == "saturation":
        saturation()
    elif args.command == "explain":
        explain()
