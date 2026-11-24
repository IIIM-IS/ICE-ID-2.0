"""ONA counting evidence for whole judgment sets: a second ICE-ID encoding (results only).

The single-judgment ONA matcher (ona_matcher.py, run_ona_full.py) gives ONA one implication per
judgment, counted on the training pairs. Here ONA also gets one implication per distinct
training judgment set, counted by the same NAL evidence rule (om.implications_from_counts:
w = training pairs whose judgment set is exactly S, w+ = the matches among them,
f = w+/w, c = min(w/(w+1), 0.99), decomposition form), next to the unchanged per-judgment
implications. ONA's own answer selection decides between the conclusions.

Why the set is one atom, not one compound term (measured with the ONA build, probes/):

* The Narsese parser is binary: ``(&, a, b, c, d)`` is read as ``(a & b)``; c and d are dropped
  without a warning (src/Narsese.c buildBinaryTree takes the copula and two arguments).
* A term is a 64-slot binary heap (COMPOUND_TERM_SIZE_MAX), so depth <= 5. In
  ``<<$1 --> C> ==> <$1 --> (C & match)>>`` the intersection C starts at depth 2 (antecedent)
  and 3 (consequent). A nested intersection of 9 judgments aborts ONA ("COMPOUND_TERM_SIZE_MAX
  too small"); 8 judgments in the one-step antecedent parse but are never matched (their atoms
  lie beyond the first UNIFICATION_DEPTH = 31 heap slots that the inverted atom index covers);
  6 work. ICE-ID judgment sets have 5-11 judgments (9-10 for 98% of the training pairs).
* So the set is named by one atom ``set_<id>`` (id from experiments/ona/full/iceid/judgment_sets.csv.gz)
  whose extension is "pairs whose judgment set is exactly S"; its counts are exactly that
  extension's training pairs. Unseen sets have no training evidence and therefore no set
  implication, exactly as a judgment without training evidence has no implication.

Answer-selection designs (ONA decides in all of them; src/Inference.c Inference_RevisionAndChoice:
disjoint stamps -> revision, overlapping stamps -> the more confident belief is kept):

* ``revision``: the pilot episode (oc.episode_lines, run_ona_pilot.DESIGN) with the set atom as one
  more observed property: ``<{px} --> set_<id>>.`` plus its implication, in the same sorted
  implication-then-observation order. Every input has its own stamp id, so ONA revises the set
  conclusion with the per-judgment conclusions.
* ``choice``: the pair is ONE observation ``<{px} --> set_<id>>.`` and its judgments follow by
  definition ``<set_<id> --> j>. %1;0.99%`` (NAL-1 deduction, src/NAL.h). Every conclusion about
  the pair then carries the observation's stamp id, so ONA treats them as overlapping evidence
  and keeps the most confident one (choice) instead of revising.
* ``choice_noset``: ``choice`` without the set implications (attribution control).

Outcome of the pilots (120 random episodes each): ``revision`` 120/120 answered, all within 1e-5 of
the clean NAL derivation; ``choice`` 85/120 and ``choice_noset`` 114/120 unanswered, because the derived
observations lose priority before the third inference step in unmodified ONA. Only ``revision`` was
run in full.

The desktop never sees a label: it gets the knowledge, the judgment-set registry, the distinct
(seed, knowledge base, set) episodes to run and an unlabelled sample of pairs for the cache check.

Subcommands (laptop: prepare, score; desktop: run, fresh-check):
  python run_ona_setcount.py prepare
  python run_ona_setcount.py run --design revision|choice|choice_noset      (desktop)
  python run_ona_setcount.py fresh-check --design ...                         (desktop)
  python run_ona_setcount.py score
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "vendor/bench"))

import ona_matcher as om  # noqa: E402
import ona_cache as oc  # noqa: E402
from run_ona_pilot import DESIGN  # noqa: E402

ICE = BASE / "experiments/ona/full/iceid"
OUT = BASE / "experiments/ona/setcount"
DESKTOP_ROOT = Path(os.environ.get("ONA_SETCOUNT_ROOT", str(Path.home() / "task-logs/ona-setcount")))
SEEDS = [131, 132, 133]
DESIGNS = ["revision", "choice", "choice_noset"]
# (pair file stem, partitions scored, knowledge base); kb as in run_ona_full.protocol_frames.
STEMS = [("controlled", ["selection", "test"], "pub"), ("publication_graph", ["selection", "test"], "pub"),
         ("candidate_graph", ["selection", "test"], "cand"), ("locked_controlled", ["locked"], "pub"),
         ("locked_candidate", ["locked"], "cand")]
DEFINITION_CONFIDENCE = om.MAX_CONFIDENCE    # analytic: set_<id> is defined by its judgments
OBSERVATION_CONFIDENCE = 0.9                 # NAR_DEFAULT_CONFIDENCE, as every pilot observation
STRUCTURAL = 0.9                             # RELIANCE (Truth_StructuralDeduction)
WORKERS_MAX = 8                              # hard cap of the desktop run


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=True) + "\n")


def set_atom(set_id) -> str:
    return om.atom(f"set_{int(set_id)}")


# ------------------------------------------------------------------ knowledge (laptop)
def set_counts(judgment_sets, labels) -> dict:
    """{set key: (w+, w)} over training pairs; keys are the frozensets themselves."""
    counts = {}
    for s, y in zip(judgment_sets, labels):
        p, n = counts.get(s, (0, 0))
        counts[s] = (p + int(y), n + 1)
    return counts


def check_marginals(counts: dict, rows: list, what: str) -> None:
    """The per-judgment counts ONA already uses must be the marginals of the set counts."""
    for r in rows:
        w = sum(n for s, (_, n) in counts.items() if r["judgment"] in s)
        wp = sum(p for s, (p, _) in counts.items() if r["judgment"] in s)
        if (w, wp) != (int(r["w"]), int(r["w_plus"])):
            raise RuntimeError(f"{what}: set counts do not reproduce {r['judgment']} ({wp}/{w} vs {r['w_plus']}/{r['w']})")
    judged = {j for s in counts for j in s}
    if judged != {r["judgment"] for r in rows}:
        raise RuntimeError(f"{what}: judgment vocabulary of the sets differs from the knowledge")


def set_knowledge(counts: dict, registry: dict) -> list:
    """Set implications by the per-judgment rule itself (om.implications_from_counts)."""
    stats = SimpleNamespace(match_counts={}, non_match_counts={})
    for s, (p, n) in counts.items():
        if s in registry:            # only sets some scored pair shows can ever be applied
            stats.match_counts[set_atom(registry[s])] = p
            stats.non_match_counts[set_atom(registry[s])] = n - p
    from run_ona_full import knowledge_rows
    return knowledge_rows(om.implications_from_counts(stats, "decomposition"))


def prepare():
    from run_ona_full import load_records
    from run_judgment_lr import candidate_training_sets
    registry_frame = pd.read_csv(ICE / "judgment_sets.csv.gz")
    registry = {frozenset(j.split()): int(i) for i, j in zip(registry_frame.judgment_set, registry_frame.judgments)}
    by_id = {i: s for s, i in registry.items()}
    single = json.loads((ICE / "knowledge.json").read_text())
    records = load_records()
    knowledge, report, needed, pairs = {}, {}, [], []
    for seed in SEEDS:
        controlled = pd.read_csv(ICE / f"pairs/seed_{seed}/controlled.csv.gz")
        train = controlled[controlled.partition.eq("train")]
        pub = set_counts([by_id[s] for s in train.judgment_set], train.label)
        check_marginals(pub, single[str(seed)]["pub"], f"{seed}/pub")
        cand_train, cand_sets = candidate_training_sets(seed, records)   # checked against the saved count model
        cand = set_counts(cand_sets, cand_train.label)
        check_marginals(cand, single[str(seed)]["cand"], f"{seed}/cand")
        knowledge[str(seed)] = {kb: single[str(seed)][kb] + set_knowledge(c, registry) for kb, c in
                                [("pub", pub), ("cand", cand)]}
        train_keys = {"pub": set(zip(train.id1, train.id2)) | set(zip(train.id2, train.id1)),
                      "cand": set(zip(cand_train.id1, cand_train.id2)) | set(zip(cand_train.id2, cand_train.id1))}
        report[str(seed)] = {kb: {"training_pairs": int(sum(n for _, n in c.values())),
                                  "training_positive": int(sum(p for p, _ in c.values())),
                                  "distinct_training_sets": len(c), "training_sets_in_registry": int(sum(s in registry for s in c)),
                                  "set_implications": int(sum(s in registry for s in c))} for kb, c in [("pub", pub), ("cand", cand)]}
        report[str(seed)]["pub"]["source"] = "controlled.csv.gz partition train (corrected_training_labels/train_pairs.csv)"
        report[str(seed)]["cand"]["source"] = "candidate-quarantined-extended/seed_<s>/train_pairs.csv via the saved count model's _preprocess"
        for stem, partitions, kb in STEMS:
            frame = pd.read_csv(ICE / f"pairs/seed_{seed}/{stem}.csv.gz", float_precision="round_trip")
            frame = frame[frame.partition.isin(partitions)]
            if set(frame.kb) != {kb}:
                raise RuntimeError(f"{seed}/{stem}: knowledge base {set(frame.kb)} is not {kb}")
            overlap = sum((a, b) in train_keys[kb] for a, b in zip(frame.id1, frame.id2))
            report[str(seed)][f"{stem}_pairs_also_in_{kb}_training"] = int(overlap)
            if overlap:
                raise RuntimeError(f"{seed}/{stem}: {overlap} scored pairs are training pairs of the counts")
            cols = ["seed", "partition", "kb", "id1", "id2", "label", "judgment_set", "ona_score", "llr_score"]
            pairs.append(frame[cols].assign(stem=stem))
            needed += [(seed, kb, int(s)) for s in frame.judgment_set.unique()]
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / "inputs/knowledge.json", knowledge)
    dump(OUT / "inputs/prepare.json", report)
    frame = pd.concat(pairs, ignore_index=True)
    frame.to_csv(OUT / "inputs/pairs.csv.gz", index=False)
    pd.DataFrame(sorted(set(needed)), columns=["seed", "kb", "judgment_set"]).to_csv(OUT / "inputs/needed.csv.gz", index=False)
    registry_frame.to_csv(OUT / "inputs/judgment_sets.csv.gz", index=False)
    # Unlabelled random sample for the cache check: ~300 pairs per (seed, stem), 4,500 in all.
    rng = np.random.default_rng(20261124)
    sample = []
    for (seed, stem), part in frame.groupby(["seed", "stem"]):
        take = rng.choice(len(part), min(300, len(part)), replace=False)
        sample.append(part.iloc[np.sort(take)][["seed", "stem", "partition", "kb", "id1", "id2", "judgment_set"]])
    pd.concat(sample, ignore_index=True).to_csv(OUT / "inputs/fresh_sample.csv.gz", index=False)
    print(json.dumps(report, indent=1))
    print(f"{len(frame)} scored pairs, {len(set(needed))} (seed, kb, set) episodes per design")


def load_knowledge(root: Path) -> dict:
    from run_ona_full import knowledge_from_rows
    raw = json.loads((root / "inputs/knowledge.json").read_text())
    return {(int(seed), kb): knowledge_from_rows(rows) for seed, kbs in raw.items() for kb, rows in kbs.items()}


def load_registry(root: Path) -> dict:
    frame = pd.read_csv(root / "inputs/judgment_sets.csv.gz")
    return {int(i): frozenset(j.split()) for i, j in zip(frame.judgment_set, frame.judgments)}


# ------------------------------------------------------------------ episodes
def choice_lines(judgments, atom_name: str, knowledge: dict, use_set: bool, key: str = oc.KEY) -> list[str]:
    """One observation of the whole set; judgments by definition; same cycles as the pilot DESIGN."""
    term = om.pair_term(key)
    lines = ["*reset", f"*currenttime={om.CLOCK_ORIGIN}", "*volume=0", f"<{term} --> {atom_name}>."]
    for j in sorted(judgments):
        lines.append(f"<{atom_name} --> {om.atom(j)}>. %1.000000000;{DEFINITION_CONFIDENCE:.9f}%")
        if j in knowledge:
            lines.append(knowledge[j].narsese)
        lines.append(str(int(DESIGN["step_cycles"])))
    if use_set and atom_name in knowledge:
        lines.append(knowledge[atom_name].narsese)
        lines.append(str(int(DESIGN["step_cycles"])))
    lines += [str(int(DESIGN["cycles"])), f"//{om.MARK}{key}", f"<{term} --> match>?"]
    return lines


def conclusion(imp: om.Implication, premise_confidence: float) -> tuple[float, float]:
    """NAL truth of <{px} --> match> through one implication (Truth_Deduction, then structural)."""
    m = imp.frequency
    c = m * imp.confidence * premise_confidence * STRUCTURAL * m
    return (1.0 - m if imp.negated else m), c


def choice_reference(judgments, atom_name: str, knowledge: dict, use_set: bool) -> dict | None:
    """Audit only: the most confident clean conclusion (ONA's choice), with any exact ties."""
    cands = []
    for j in sorted(judgments):
        if j in knowledge:
            cands.append((f"judgment:{j}",) + conclusion(knowledge[j], OBSERVATION_CONFIDENCE * DEFINITION_CONFIDENCE))
    if use_set and atom_name in knowledge:
        cands.append(("set",) + conclusion(knowledge[atom_name], OBSERVATION_CONFIDENCE))
    if not cands:
        return None
    top = max(c for _, _, c in cands)
    tied = [x for x in cands if abs(x[2] - top) <= 1e-12]
    f, c = tied[0][1], tied[0][2]
    return {"frequency": f, "confidence": c, "expectation": c * (f - 0.5) + 0.5, "source": tied[0][0],
            "tied": [{"source": s, "frequency": tf, "confidence": tc} for s, tf, tc in tied]}


def input_roles(lines: list[str]) -> dict:
    """Stamp id -> input sentence (ONA numbers input sentences 1, 2, ... after *reset)."""
    roles, n = {}, 0
    for line in lines:
        if line.startswith("<"):
            n += 1
            roles[n] = line
    return roles


def classify(stamp: str, roles: dict) -> str:
    """Which conclusion an answer is: from the input sentences its evidential base names."""
    ids = [int(x) for x in stamp.split(",") if x]
    kinds = set()
    for i in ids:
        line = roles[i]
        if "==>" in line:
            antecedent = line.split(" --> ", 1)[1].split(">", 1)[0]
            kinds.add("set" if antecedent.startswith("set_") else "judgment")
    if not kinds:
        return "no_implication"
    return kinds.pop() if len(kinds) == 1 else "mixed"


class Store(oc.EpisodeStore):
    """oc.EpisodeStore for any episode text: same marker, batching, archive and audit."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.roles = {}

    def request_text(self, lines: list[str], judgments, n_implications: int, reference) -> str:
        text = "\n".join(lines) + "\n"
        digest = oc.sha(text)
        self.roles[digest] = input_roles(lines)
        if digest not in self.records and digest not in self.pending:
            self.pending[digest] = {"text": text, "judgments": " ".join(sorted(judgments)),
                                    "n_judgments": len(set(judgments)), "n_implications": n_implications,
                                    "reference": reference}
        return digest


def episode_for(design: str, judgments, set_id: int, knowledge: dict, key: str = oc.KEY):
    """(lines, reference) of one pair's episode under ``design``; never consults a label."""
    name = set_atom(set_id)
    if design == "revision":
        present = set(judgments) | {name}
        return oc.episode_lines(present, knowledge, key=key), om.nal_reference(present, knowledge)
    use_set = design == "choice"
    return (choice_lines(judgments, name, knowledge, use_set, key),
            choice_reference(judgments, name, knowledge, use_set))


# ------------------------------------------------------------------ desktop runtime
def desktop_runtime(root: Path) -> dict:
    """Desktop NAR built from the recorded commit; accepted only after the byte-for-byte check."""
    nar = root / "ona-src/NAR"
    repro = json.loads((root / "repro/report.json").read_text())
    digest = om.sha256_file(nar)
    if digest != repro["desktop_nar_sha256"] or repro["identical_episodes"] != repro["episodes"] or repro["episodes"] < 200:
        raise RuntimeError("Desktop NAR is not the one that reproduced the laptop NAR byte for byte")
    om.ONA_BINARY = nar
    oc.WORKERS = WORKERS_MAX          # the desktop cap (ona_cache's own default is 4 for the laptop)
    return {"binary": str(nar), "sha256": digest, "laptop_sha256": om.ONA_SHA256, "commit": om.ONA_COMMIT,
            "source": repro["source"], "reproduced_laptop_episodes": repro["identical_episodes"],
            "reproduced_laptop_batches": repro["identical_batches"], "design": DESIGN,
            "max_confidence": om.MAX_CONFIDENCE, "clock_origin": om.CLOCK_ORIGIN}


def run(design: str, workers: int, limit: int | None = None, tag: str = ""):
    root = DESKTOP_ROOT
    runtime = desktop_runtime(root)
    knowledge, registry = load_knowledge(root), load_registry(root)
    needed = pd.read_csv(root / "inputs/needed.csv.gz")
    if limit:
        needed = needed.sample(n=min(limit, len(needed)), random_state=20261124).sort_values(["seed", "kb", "judgment_set"])
    store = Store(root / f"runs/{design}{tag}")
    digests = []
    for seed, kb, s in needed.itertuples(index=False):
        lines, reference = episode_for(design, registry[s], s, knowledge[(seed, kb)])
        n_imp = sum(1 for line in lines if "==>" in line)
        digests.append(store.request_text(lines, registry[s], n_imp, reference))
    print(f"{design}{tag}: {len(needed)} (seed, kb, set) episodes, {len(set(digests))} distinct inputs, "
          f"{len(store.pending)} to run", flush=True)
    ran = store.run(workers=workers)
    rows = []
    for (seed, kb, s), d in zip(needed.itertuples(index=False), digests):
        rec = store.get(d)
        rows.append({"seed": seed, "kb": kb, "judgment_set": s, "input_sha256": d, "episode_id": rec["episode_id"],
                     "answered": rec["answered"], "expectation": rec["expectation"], "frequency": rec["frequency"],
                     "confidence": rec["confidence"], "stamp": rec["stamp"], "answer_line": rec["answer_line"],
                     "anomaly": rec["anomaly"], "dev_expectation": rec["dev_expectation"],
                     "answer_kind": classify(rec["stamp"], store.roles[d]) if rec["answered"] else "unanswered"})
    frame = pd.DataFrame(rows)
    frame.to_csv(store.root / "answers.csv.gz", index=False)
    store.export()
    timing = {"design": design, "runtime": runtime, "new_episodes": ran["new_episodes"], "ona_seconds": ran["seconds"],
              "workers": workers, "episodes": len(frame), "distinct_inputs": int(frame.input_sha256.nunique()),
              "answered": int(frame.answered.sum()), "anomalies": frame.anomaly.fillna("").value_counts().to_dict(),
              "answer_kinds": frame.answer_kind.value_counts().to_dict()}
    previous = store.root / "run.json"
    history = json.loads(previous.read_text())["history"] if previous.exists() else []
    dump(previous, {**timing, "history": history + [{k: timing[k] for k in ["new_episodes", "ona_seconds", "workers"]}
                                                     | {"finished": time.strftime("%Y-%m-%d %H:%M:%S")}]})
    print(json.dumps({k: v for k, v in timing.items() if k != "runtime"}, indent=1, default=str))


def fresh_check(design: str, own_process: int = 64, extra: int = 100):
    """Sampled pairs re-run as pilot-style per-pair episodes; full output segment vs the cached one."""
    root = DESKTOP_ROOT
    runtime = desktop_runtime(root)
    knowledge, registry = load_knowledge(root), load_registry(root)
    store = Store(root / f"runs/{design}")
    sample = pd.read_csv(root / "inputs/fresh_sample.csv.gz")
    answers = pd.read_csv(store.root / "answers.csv.gz")
    rng = np.random.default_rng(20261126)
    picked = [(f"{r.seed}/{r.stem}/{r.partition}/{r.id1}-{r.id2}", int(r.seed), r.kb, int(r.judgment_set))
              for r in sample.itertuples(index=False)]
    covered = {(s, k, j) for _, s, k, j in picked}
    rest = answers[[(s, k, j) not in covered for s, k, j in zip(answers.seed, answers.kb, answers.judgment_set)]]
    for i in rng.choice(len(rest), min(extra, len(rest)), replace=False):
        r = rest.iloc[int(i)]
        picked.append((f"{r.seed}/distinct-episode", int(r.seed), r.kb, int(r.judgment_set)))
    episodes, meta = [], []
    for n, (label, seed, kb, s) in enumerate(picked):
        key = f"v{n}"
        lines, _ = episode_for(design, registry[s], s, knowledge[(seed, kb)], key=key)
        cached_lines, _ = episode_for(design, registry[s], s, knowledge[(seed, kb)])
        digest = oc.sha("\n".join(cached_lines) + "\n")
        if digest not in store.records:
            raise RuntimeError(f"{label}: no cached episode for this pair's input")
        episodes.append((key, lines))
        meta.append((label, key, digest))
    archive = store.root / "fresh_check"
    batched, seconds = oc.run_marked(episodes, om.MARK, archive, "fresh", workers=WORKERS_MAX)
    alone, alone_seconds = oc.run_marked(episodes[:own_process], om.MARK, archive, "alone", workers=WORKERS_MAX, batch_size=1)
    rows = []
    for label, key, digest in meta:
        cached = store.segment_of(digest)
        fresh = oc.renamed(batched[key][0], key)
        row = {"pair": label, "input_sha256": digest, "episode_id": store.get(digest)["episode_id"],
               "fresh_answer_line": batched[key][0].rstrip("\n").splitlines()[-1],
               "cached_answer_line": store.get(digest)["answer_line"], "segment_identical": fresh == cached,
               "answer_line_identical": fresh.rstrip("\n").splitlines()[-1] == cached.rstrip("\n").splitlines()[-1]}
        if key in alone:
            row["own_process_segment_identical"] = oc.renamed(alone[key][0], key) == cached
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(archive / "fresh_vs_cached.csv.gz", index=False)
    report = {"design": design, "runtime_sha256": runtime["sha256"], "pairs": len(frame),
              "random_pairs": len(sample), "distinct_episode_extra": len(picked) - len(sample),
              "distinct_episodes_covered": int(frame.episode_id.nunique()), "episodes_in_store": len(store.records),
              "segment_identical": int(frame.segment_identical.sum()),
              "answer_line_identical": int(frame.answer_line_identical.sum()),
              "own_process_checked": int(frame.own_process_segment_identical.notna().sum()),
              "own_process_identical": int(frame.own_process_segment_identical.fillna(False).astype(bool).sum()),
              "batched_seconds": seconds, "own_process_seconds": alone_seconds,
              "comparison": "complete per-episode ONA output, per-pair term {pv<n>} and marker renamed to the constant ones; "
                            f"100 per NAR process; the first {own_process} also alone in their own NAR process"}
    report["all_identical"] = bool(report["segment_identical"] == report["answer_line_identical"] == len(frame)
                                   and report["own_process_identical"] == report["own_process_checked"])
    dump(archive / "report.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "run", "fresh-check", "score"])
    parser.add_argument("--design", choices=DESIGNS)
    parser.add_argument("--workers", type=int, default=WORKERS_MAX)
    parser.add_argument("--limit", type=int, default=None, help="pilot: number of (seed, kb, set) episodes")
    parser.add_argument("--tag", default="", help="suffix of the run folder (pilots)")
    args = parser.parse_args()
    if args.workers > WORKERS_MAX:
        raise SystemExit(f"At most {WORKERS_MAX} ONA processes")
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run(args.design, args.workers, args.limit, args.tag)
    elif args.command == "fresh-check":
        fresh_check(args.design)
    else:
        from run_ona_setcount_metrics import main as score
        score()
