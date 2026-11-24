"""Exact-input episode cache for the ONA matcher (ona_matcher.py; unmodified ONA binary).

Every ONA scoring episode starts with ``*reset`` (Shell_NARInit -> NAR_INIT: memory, event
ids/stamps, atoms, cycle state and clock), so its output is a function of its Narsese input
alone. The pilot names each pair by its own term ({p<key>}); here the pair term is the constant
{px}, so pairs with the same judgment set and the same implications get byte-identical
inputs (ona_matcher.episode with key "x", pilot DESIGN unchanged). Each distinct input is run
once and its answer is used for every such pair. The only per-episode difference inside a
batch is the comment line that delimits the output (``//ONA_EP|<id>``): ONA prints comments
and returns without touching memory (src/Shell.c, Shell_ProcessInput).

``fresh_check`` re-runs sampled pairs as pilot-style per-pair episodes (pair-specific term
and marker, reset-separated, plus some in their own NAR process) and compares the complete
per-episode ONA output with the cached one after renaming the pair term.

No score is ever filled: an unanswered episode keeps answered=False and NaN truth values;
an answer that deviates from the clean NAL derivation (ona_matcher.nal_reference) by more
than TOLERANCE in frequency, confidence or expectation is flagged as an anomaly, not
replaced.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import ona_matcher as om
from run_ona_pilot import DESIGN

KEY = "x"
TERM = om.pair_term(KEY)                 # {px}
MARKER = f"//{om.MARK}{KEY}"             # the marker line ona_matcher.episode writes
EP = "ONA_EP|"
TOLERANCE = 1e-5
WORKERS = 4                              # hard cap: at most four parallel ONA processes
BATCH = 100                              # episodes per NAR process (pilot DESIGN batch_size)


def episode_lines(judgments, knowledge, key=KEY, volume=0):
    """The pilot episode (ona_matcher.episode with the frozen pilot DESIGN)."""
    return om.episode(key, judgments, knowledge, DESIGN["cycles"], volume, DESIGN["knowledge_scope"],
                      DESIGN["step_cycles"], DESIGN["order"])


def canonical(judgments, knowledge) -> str:
    return "\n".join(episode_lines(judgments, knowledge)) + "\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def segments(stdout: str, marker_prefix: str) -> dict:
    """Split one NAR transcript into per-episode output segments keyed by their marker.

    A segment is every output line after the previous episode's answer up to and including
    this episode's ``Answer:`` line. The trailing process statistics are returned under None.
    """
    out, current, pending = {}, [], None
    for line in stdout.splitlines():
        current.append(line)
        if line.startswith("Comment: " + marker_prefix):
            if pending is not None:
                raise RuntimeError(f"Episode {pending} has no answer line")
            pending = line[len("Comment: " + marker_prefix):]
            if pending in out:
                raise RuntimeError(f"Duplicate episode marker {pending}")
        elif line.startswith("Answer:") and pending is not None:
            out[pending] = "\n".join(current) + "\n"
            current, pending = [], None
    if pending is not None:
        raise RuntimeError(f"Episode {pending} has no answer line")
    out[None] = "\n".join(current) + ("\n" if current else "")
    return out


def parse_answer(segment: str, term: str) -> dict | None:
    """ONA's printed answer of one segment (the last line); None if ONA answered None."""
    line = segment.rstrip("\n").splitlines()[-1]
    if line == "Answer: None.":
        return None
    match = om.ANSWER.fullmatch(line)
    if match is None or match.group("term") != f"<{term} --> match>":
        raise RuntimeError(f"Unexpected ONA answer line: {line}")
    f, c = float(match.group("f")), float(match.group("c"))
    return {"frequency": f, "confidence": c, "expectation": c * (f - 0.5) + 0.5,
            "stamp": match.group("stamp"), "line": line}


def run_marked(episodes: list, marker_prefix: str, archive: Path | None, tag: str,
               workers: int = WORKERS, batch_size: int = BATCH) -> tuple[dict, float]:
    """Run [(marker_key, lines)] in reset-separated NAR processes; returns ({key: segment}, seconds).

    ``lines`` must already carry the ``//<marker_prefix><key>`` comment line.
    """
    if not 1 <= workers <= WORKERS:
        raise ValueError("At most four parallel ONA processes")
    batches = [episodes[i:i + batch_size] for i in range(0, len(episodes), batch_size)]
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)

    def work(index):
        batch = batches[index]
        lines = [line for _, episode in batch for line in episode]
        stdout = om.run_batch(lines)
        if archive is not None:
            with gzip.open(archive / f"{tag}_{index:05d}.nal.gz", "wt") as stream:
                stream.write("\n".join(lines) + "\n")
            with gzip.open(archive / f"{tag}_{index:05d}.out.gz", "wt") as stream:
                stream.write(stdout)
        found = segments(stdout, marker_prefix)
        found.pop(None)
        if set(found) != {k for k, _ in batch}:
            raise RuntimeError(f"Batch {tag}_{index}: {len(found)} answers for {len(batch)} episodes")
        return index, found

    start = time.time()
    result = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, found in pool.map(work, range(len(batches))):
            for key, segment in found.items():
                result[key] = (segment, f"{tag}_{index:05d}")
    return result, time.time() - start


def audit(answer: dict | None, reference: dict | None) -> dict:
    """Coverage and fidelity flags for one ONA answer against the clean NAL derivation."""
    row = {"answered": answer is not None,
           "frequency": answer["frequency"] if answer else float("nan"),
           "confidence": answer["confidence"] if answer else float("nan"),
           "expectation": answer["expectation"] if answer else float("nan"),
           "stamp": answer["stamp"] if answer else "", "answer_line": answer["line"] if answer else "Answer: None.",
           "nal_frequency": reference["frequency"] if reference else float("nan"),
           "nal_confidence": reference["confidence"] if reference else float("nan"),
           "nal_expectation": reference["expectation"] if reference else float("nan")}
    if answer is None:
        row.update(dev_frequency=float("nan"), dev_confidence=float("nan"), dev_expectation=float("nan"),
                   anomaly="unanswered" if reference is not None else "unanswered_no_knowledge")
        return row
    if reference is None:
        row.update(dev_frequency=float("nan"), dev_confidence=float("nan"), dev_expectation=float("nan"),
                   anomaly="answered_without_reference")
        return row
    dev = {k: abs(answer[k] - reference[k]) for k in ["frequency", "confidence", "expectation"]}
    row.update({f"dev_{k}": v for k, v in dev.items()})
    row["anomaly"] = "deviation" if max(dev.values()) > TOLERANCE else ""
    return row


class EpisodeStore:
    """Distinct episode inputs, their ONA outputs and audit rows; append-only on disk.

    Every run appends its rows to root/episodes.jsonl and keeps each NAR process's input and
    output in root/batches/ (gzip), so nothing is lost if a later step fails.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "episodes.jsonl"
        self.records, self.pending, self._segments = {}, {}, {}
        if self.index.exists():
            for line in self.index.read_text().splitlines():
                row = json.loads(line)
                self.records[row["input_sha256"]] = row

    @property
    def records_by_id(self) -> dict:
        return {row["episode_id"]: row for row in self.records.values()}

    def request(self, judgments, knowledge) -> str:
        text = canonical(judgments, knowledge)
        digest = sha(text)
        if digest not in self.records and digest not in self.pending:
            present = sorted(j for j in judgments if j in knowledge)
            self.pending[digest] = {"text": text, "judgments": " ".join(sorted(judgments)),
                                    "n_judgments": len(set(judgments)), "n_implications": len(present),
                                    "reference": om.nal_reference(judgments, knowledge)}
        return digest

    def run(self, workers: int = WORKERS, batch_size: int = BATCH) -> dict:
        if not self.pending:
            return {"new_episodes": 0, "seconds": 0.0}
        digests = sorted(self.pending)
        first = len(self.records)
        ids = {d: first + i for i, d in enumerate(digests)}
        episodes = []
        for d in digests:
            lines = self.pending[d]["text"].rstrip("\n").split("\n")
            if lines[-2] != MARKER:
                raise RuntimeError("Episode input lacks the constant marker")
            episodes.append((str(ids[d]), lines[:-2] + [f"//{EP}{ids[d]}"] + lines[-1:]))
        tag = f"run{int(time.time())}"
        found, seconds = run_marked(episodes, EP, self.root / "batches", tag, workers, batch_size)
        with self.index.open("a") as stream:
            for d in digests:
                meta = self.pending[d]
                segment, batch = found[str(ids[d])]
                answer = parse_answer(segment, TERM)
                row = {"episode_id": ids[d], "input_sha256": d, "batch": batch,
                       "judgments": meta["judgments"], "n_judgments": meta["n_judgments"],
                       "n_implications": meta["n_implications"],
                       "segment_sha256": sha(segment.replace(f"Comment: {EP}{ids[d]}\n", f"Comment: {om.MARK}{KEY}\n")),
                       **audit(answer, meta["reference"])}
                self.records[d] = row
                stream.write(json.dumps(row, allow_nan=True) + "\n")
        done = {"new_episodes": len(digests), "seconds": seconds}
        self.pending = {}
        return done

    def get(self, digest: str) -> dict:
        return self.records[digest]

    def segment_of(self, digest: str) -> str:
        """Re-read the archived output segment of a cached episode (normalised marker)."""
        row = self.records[digest]
        if row["batch"] not in self._segments:
            with gzip.open(self.root / "batches" / f"{row['batch']}.out.gz", "rt") as stream:
                self._segments[row["batch"]] = segments(stream.read(), EP)
        return self._segments[row["batch"]][str(row["episode_id"])].replace(f"Comment: {EP}{row['episode_id']}\n",
                                                     f"Comment: {om.MARK}{KEY}\n")

    def export(self):
        """episodes.csv.gz (one row per distinct input) and inputs.nal.gz (canonical inputs)."""
        import pandas as pd
        frame = pd.DataFrame(sorted(self.records.values(), key=lambda r: r["episode_id"]))
        frame.to_csv(self.root / "episodes.csv.gz", index=False)
        return frame


def renamed(segment: str, key: str) -> str:
    """A per-pair episode's output with its own pair term and marker replaced by the constant ones."""
    out = segment.replace(om.pair_term(key), TERM)
    return out.replace(f"Comment: {om.MARK}{key}\n", f"Comment: {om.MARK}{KEY}\n")


def fresh_check(samples: list, store: EpisodeStore, archive: Path, own_process: int = 64) -> tuple[list, dict]:
    """samples: [(label, judgments, knowledge)] -> rows comparing per-pair with cached output.

    Each sampled pair is scored the way the pilot scored every pair: its own pair term
    {pv<n>} and marker, one reset episode per pair, 100 per NAR process; the first
    ``own_process`` of them additionally each in a NAR process of its own.
    """
    episodes, keys, digests = [], [], []
    for n, (_, judgments, knowledge) in enumerate(samples):
        key = f"v{n}"
        keys.append(key)
        episodes.append((key, episode_lines(judgments, knowledge, key=key)))
        digests.append(sha(canonical(judgments, knowledge)))
    batched, seconds = run_marked(episodes, om.MARK, archive, "fresh")
    alone, alone_seconds = {}, 0.0
    if own_process:
        alone, alone_seconds = run_marked(episodes[:own_process], om.MARK, archive, "alone", batch_size=1)
    rows = []
    for n, (label, judgments, _) in enumerate(samples):
        key, digest = keys[n], digests[n]
        cached = store.segment_of(digest)
        fresh = renamed(batched[key][0], key)
        row = {"sample": n, "protocol": label, "judgments": " ".join(sorted(judgments)), "input_sha256": digest,
               "episode_id": store.get(digest)["episode_id"],
               "fresh_answer_line": batched[key][0].rstrip("\n").splitlines()[-1],
               "cached_answer_line": store.get(digest)["answer_line"],
               "segment_identical": fresh == cached,
               "answer_line_identical": fresh.rstrip("\n").splitlines()[-1] == cached.rstrip("\n").splitlines()[-1]}
        if key in alone:
            row["own_process_segment_identical"] = renamed(alone[key][0], key) == cached
        rows.append(row)
    return rows, {"batched_seconds": seconds, "own_process_seconds": alone_seconds}


def saturation(digests: list, store: EpisodeStore, archive: Path, workers: int = WORKERS) -> list:
    """Concept-memory statistics right after the answer (``*stats`` appended after the question).

    The question is answered when it is input, so the statistics line cannot change the answer;
    the answer line is still compared with the scored one. Inputs are re-read from the batch
    archive and checked against their cache key.
    """
    texts = {}
    episodes = []
    digests = list(dict.fromkeys(digests))
    for d in digests:
        row = store.get(d)
        if row["batch"] not in texts:
            with gzip.open(store.root / "batches" / f"{row['batch']}.nal.gz", "rt") as stream:
                texts[row["batch"]] = stream.read()
        block = re.search(rf"\*reset\n(?:(?!\*reset\n).)*?//{re.escape(EP)}{row['episode_id']}\n[^\n]*\n",
                          texts[row["batch"]], re.S)
        if block is None:
            raise RuntimeError(f"Episode {row['episode_id']} input not found in its batch archive")
        lines = block.group(0).rstrip("\n").split("\n")
        if sha("\n".join(lines[:-2] + [MARKER] + lines[-1:]) + "\n") != d:
            raise RuntimeError("Archived episode input differs from its cache key")
        episodes.append((row["episode_id"], lines + ["*stats"]))
    batches = [episodes[i:i + BATCH] for i in range(0, len(episodes), BATCH)]
    archive.mkdir(parents=True, exist_ok=True)

    def work(index):
        lines = [line for _, episode in batches[index] for line in episode]
        stdout = om.run_batch(lines)
        with gzip.open(archive / f"stats_{index:05d}.out.gz", "wt") as stream:
            stream.write(stdout)
        found, current, answer = {}, None, {}
        for line in stdout.splitlines():
            if line.startswith("Comment: " + EP):
                current = int(line[len("Comment: " + EP):])
            elif line.startswith("Answer:") and current is not None:
                answer[current] = line
            elif current is not None and current in answer and current not in found:
                m = re.match(r"total concepts:\s*(\d+)", line)
                if m:
                    found[current] = int(m.group(1))
        return answer, found

    rows, by_id = [], store.records_by_id
    with ThreadPoolExecutor(max_workers=min(workers, WORKERS)) as pool:
        for answer, found in pool.map(work, range(len(batches))):
            for eid, concepts in found.items():
                row = by_id[eid]
                rows.append({"episode_id": eid, "n_judgments": row["n_judgments"],
                             "n_implications": row["n_implications"], "anomaly": row["anomaly"],
                             "answer_identical": answer[eid] == row["answer_line"],
                             "total_concepts": concepts, "concepts_max": 16384,
                             "saturated": concepts >= 16384})
    if len(rows) != len(digests):
        raise RuntimeError(f"Statistics for {len(rows)} of {len(digests)} episodes")
    return rows
