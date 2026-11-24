"""Entity matching in which OpenNARS for Applications (ONA) does the inference.

The pair judgments are exactly those of the published NARS/LLR model
(bench.models.nars.NarsModel._preprocess, imported, not re-implemented), so only the
inference engine differs from the additive log-likelihood-ratio (LLR) scorer.

Knowledge, per judgment J, is one eternal implication whose truth value follows the NAL
evidence definition on the training pairs the LLR model counted (its JudgmentStatistics):
w = training pairs showing J, w+ = the matches among them, c = w/(w+k), k = 1, capped at
ONA's MAX_CONFIDENCE (0.99, src/Config.h). Chosen form ("decomposition"):

    <<$1 --> J> ==> <$1 --> (J & match)>>.   %w+/w;c%    when w+/w >= 0.5
    <<$1 --> J> ==> <$1 --> (J - match)>>.   %w-/w;c%    when w+/w <  0.5

i.e. "a pair showing J is a J-and-match case" (extensional intersection) or "a J-but-not-match
case" (extensional difference); the stated frequency is the share of the J training pairs
in that set, so it is >= 0.5 and NAL deduction (Truth_Deduction, c = f1 f2 c1 c2) keeps the
evidence. ONA then decomposes the conclusion with its NAL-3 structural rules
(src/NAL.h: (M --> (S & P)) |- (M --> P) Truth_StructuralDeduction;
(M --> (B - S)) |- (M --> S) Truth_StructuralDeductionNegated) into <{pair} --> match> and
revises all of these (Truth_Revision) in the one concept the question is asked about.

Rejected form ("negation"): <<$1 --> J> ==> (! <$1 --> match>)>. for w+/w < 0.5. All such
conclusions share the single concept (! <{pair} --> match>); once it holds a revised belief,
ONA admits neither further derived events of lower confidence nor the revised belief itself
to the cycling queue (Memory_addCyclingEvent, Memory_containsBeliefOrGoal), so the negation
rule (! A) |- A is applied to at most the first one or two negative conclusions and the rest
of the negative evidence never reaches <{pair} --> match> (shown in the pilot design study).
The decomposition form gives every judgment its own intermediate concept instead.

Scoring: every pair is its own episode in a freshly reset ONA memory (``*reset``), so test
pairs never share memory with each other or with any label. The episode contains, per
judgment, its implication directly followed by the judgment <{pair} --> J>. (eternal, default
truth 1.0;0.9) and a few inference cycles; then more cycles and the question
<{pair} --> match>?. ONA derives everything itself (variable deduction in
Cycle_SpecialInferences, the structural rules of NAL.h, revision in
Memory_ProcessNewBeliefEvent) and answers from its concept memory (NAR_AddInputNarsese2).
The score is the expectation c (f - 0.5) + 0.5 of the answer ONA prints. A pair that ONA does
not answer gets no score (NaN); it is counted, never filled by a formula.

The unmodified binary is checked against its recorded hash and source commit. After every
``*reset`` the clock is set with ``*currenttime=CLOCK_ORIGIN``, a precaution: ONA's adaptive
concept priority threshold uses Stats_countConceptsMatchedTotal / currentTime and that
statistic is not cleared by ``*reset``. With the clock origin the quotient stays 0 for any
history. (Measured in the pilot clock check: 150 reset-separated episodes give identical
answers with and without it, so it is not needed at pilot scale; position, fresh-process and
volume independence are checked in the design study.)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import os
from pathlib import Path
import re
import subprocess
import time

ONA_DIR = Path(os.environ.get("ONA_DIR", "OpenNARS-for-Applications"))  # unmodified build of ONA_COMMIT (see README)
ONA_BINARY = ONA_DIR / "NAR"
ONA_COMMIT = "f14cfce95ce9ffc2eb1a2ac52ac827bc25d528ee"
ONA_SHA256 = "d006f11b10e01c146e61c27dfb891ebd924f8c56433ea7886b83ff6cfb7a6187"
MAX_CONFIDENCE = 0.99          # src/Config.h MAX_CONFIDENCE
EVIDENTIAL_HORIZON = 1.0       # src/Config.h TRUTH_EVIDENTAL_HORIZON_INITIAL
ATOM_LEN_MAX = 31              # src/Config.h ATOMIC_TERM_LEN_MAX - 1 (longer names are truncated by ONA)
CLOCK_ORIGIN = 1_000_000_000
ONA_RSS_GIB = 1.0              # measured peak RSS of one NAR shell process (static arrays)
ATOM = re.compile(r"[a-z][a-z0-9_]*")
ANSWER = re.compile(
    r"Answer: (?P<term>.+?)\. creationTime=(?P<creation>-?\d+) Stamp=\[(?P<stamp>[\d,]*)\] "
    r"Truth: frequency=(?P<f>\d+\.\d+), confidence=(?P<c>\d+\.\d+)")
MARK = "ONA_PAIR|"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime() -> dict:
    """The unmodified ONA build: binary hash, source commit and a clean src/ tree."""
    digest = sha256_file(ONA_BINARY)
    if digest != ONA_SHA256:
        raise RuntimeError(f"ONA binary hash {digest} differs from the recorded {ONA_SHA256}")
    commit = subprocess.run(["git", "-C", str(ONA_DIR), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ONA_DIR), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if commit != ONA_COMMIT or dirty:
        raise RuntimeError(f"ONA source is not the recorded unmodified commit: {commit} dirty={bool(dirty)}")
    return {"binary": str(ONA_BINARY), "sha256": digest, "commit": commit, "src_clean": True,
            "mode": "NAR shell (stdin/stdout)", "max_confidence": MAX_CONFIDENCE,
            "evidential_horizon": EVIDENTIAL_HORIZON, "clock_origin": CLOCK_ORIGIN,
            "random_seed": "compiled default mysrand(666) in src/main.c; unused without operations"}


def atom(name: str) -> str:
    if not ATOM.fullmatch(name) or len(name) > ATOM_LEN_MAX or name == "match":
        raise ValueError(f"Judgment {name!r} is not a valid ONA atom; refusing to rename it silently")
    return name


FORMS = ("decomposition", "negation")


@dataclass(frozen=True)
class Implication:
    judgment: str
    positive: int          # training matches showing the judgment (w+)
    total: int             # training pairs showing the judgment (w)
    frequency: float       # share of the stated consequent (>= 0.5)
    confidence: float
    negated: bool          # consequent states non-match
    form: str = "decomposition"

    @property
    def narsese(self) -> str:
        j = atom(self.judgment)
        if self.form == "decomposition":
            consequent = f"<$1 --> ({j} - match)>" if self.negated else f"<$1 --> ({j} & match)>"
        elif self.form == "negation":
            consequent = "(! <$1 --> match>)" if self.negated else "<$1 --> match>"
        else:
            raise ValueError(self.form)
        return f"<<$1 --> {j}> ==> {consequent}>. %{self.frequency:.9f};{self.confidence:.9f}%"


def implications_from_counts(stats, form: str = "decomposition", k: float = EVIDENTIAL_HORIZON,
                             cap: float = MAX_CONFIDENCE) -> dict:
    """NAL evidence (w+ = matches with J, w = pairs with J) from the LLR model's own counts."""
    if form not in FORMS:
        raise ValueError(form)
    knowledge = {}
    for judgment in sorted(set(stats.match_counts) | set(stats.non_match_counts)):
        positive = int(stats.match_counts.get(judgment, 0))
        total = positive + int(stats.non_match_counts.get(judgment, 0))
        if total == 0:
            continue
        f = positive / total
        c = min(total / (total + k), cap)
        knowledge[judgment] = Implication(judgment, positive, total, f if f >= 0.5 else 1.0 - f, c,
                                          f < 0.5, form)
    return knowledge


def pair_term(key: str) -> str:
    return "{" + atom("p" + key) + "}"


def episode(key: str, judgments, knowledge: dict, cycles: int, volume: int = 0,
            knowledge_scope: str = "present", step_cycles: int = 0, order: str = "sorted",
            group_cycles: int = 0) -> list[str]:
    """Narsese for one pair, starting from a reset memory.

    knowledge_scope 'present' inputs the implications whose antecedent the pair shows;
    'all' inputs the whole learned knowledge base. ``step_cycles`` inference cycles follow
    each judgment; ``cycles`` follow the last one, before the question. ``order`` 'sorted'
    inputs judgments alphabetically, 'positive_first' those whose implication concludes
    match before those concluding its negation (alphabetical within each group).
    """
    term = pair_term(key)
    judgments = sorted(judgments)
    if order == "positive_first":
        judgments.sort(key=lambda j: (j in knowledge and knowledge[j].negated, j))
    elif order == "negative_first":
        judgments.sort(key=lambda j: (not (j in knowledge and knowledge[j].negated), j))
    elif order != "sorted":
        raise ValueError(order)
    lines = ["*reset", f"*currenttime={CLOCK_ORIGIN}", f"*volume={volume}"]
    if knowledge_scope == "all":
        lines += [knowledge[j].narsese for j in sorted(set(knowledge) - set(judgments))]
    elif knowledge_scope != "present":
        raise ValueError(knowledge_scope)
    # Each implication directly precedes its judgment: when ONA selects the judgment event the
    # implication concept is still fresh, so the deduced conclusion is not out-prioritised by
    # the compositions ONA forms among the pair's judgments (Cycle_Inference, NAL_DerivedEvent
    # priority = concept priority * parent priority * expectation).
    previous_negated = None
    for j in judgments:
        negated = j in knowledge and knowledge[j].negated
        if group_cycles and previous_negated is not None and negated != previous_negated:
            lines.append(str(int(group_cycles)))
        previous_negated = negated
        if j in knowledge:
            lines.append(knowledge[j].narsese)
        lines.append(f"<{term} --> {atom(j)}>.")
        if step_cycles:
            lines.append(str(int(step_cycles)))
    if cycles:
        lines.append(str(int(cycles)))
    lines += [f"//{MARK}{key}", f"<{term} --> match>?"]
    return lines


def parse_answers(stdout: str) -> dict:
    """Map each episode key to ONA's printed answer, checking the answered term."""
    answers, pending = {}, None
    for line in stdout.splitlines():
        if line.startswith("Comment: " + MARK):
            if pending is not None:
                raise RuntimeError(f"Episode {pending} has no answer line")
            pending = line[len("Comment: " + MARK):]
            if pending in answers:
                raise RuntimeError(f"Duplicate episode key {pending}")
        elif line.startswith("Answer:") and pending is not None:
            if line == "Answer: None.":
                answers[pending] = None
            else:
                match = ANSWER.fullmatch(line)
                expected = f"<{pair_term(pending)} --> match>"
                if match is None or match.group("term") != expected:
                    raise RuntimeError(f"Unexpected ONA answer for {pending}: {line}")
                f, c = float(match.group("f")), float(match.group("c"))
                answers[pending] = {"frequency": f, "confidence": c, "expectation": c * (f - 0.5) + 0.5,
                                    "stamp": match.group("stamp"), "line": line}
            pending = None
    if pending is not None:
        raise RuntimeError(f"Episode {pending} has no answer line")
    return answers


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2
    raise RuntimeError("MemAvailable missing")


def run_batch(lines: list[str], timeout: float = 3600, min_free_gib: float = 4.0) -> str:
    """One unmodified NAR shell process over a batch of reset-separated episodes."""
    waited = 0
    while available_gib() - ONA_RSS_GIB < min_free_gib:
        if waited > 1800:
            raise RuntimeError(f"Memory stayed below {min_free_gib} GiB free for 30 min")
        time.sleep(10)
        waited += 10
    done = subprocess.run([str(ONA_BINARY), "shell"], input="\n".join(lines) + "\n", text=True,
                          capture_output=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"ONA exited with {done.returncode}: {done.stderr[-2000:]}")
    return done.stdout


def score_episodes(episodes: dict, batch_size: int = 200, workers: int = 3, archive: Path | None = None):
    """Run {key: lines} episodes in batches; returns ({key: answer or None}, seconds).

    Every batch's Narsese input and ONA output are written to ``archive`` (gzip) when given.
    """
    if not 1 <= workers <= 4:
        raise ValueError("At most four parallel ONA processes")
    keys = list(episodes)
    batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)

    def work(index):
        lines = [line for key in batches[index] for line in episodes[key]]
        stdout = run_batch(lines)
        if archive is not None:
            with gzip.open(archive / f"batch_{index:05d}.nal.gz", "wt") as stream:
                stream.write("\n".join(lines) + "\n")
            with gzip.open(archive / f"batch_{index:05d}.out.gz", "wt") as stream:
                stream.write(stdout)
        answers = parse_answers(stdout)
        if set(answers) != set(batches[index]):
            raise RuntimeError(f"Batch {index}: answers for {len(answers)} of {len(batches[index])} episodes")
        return answers

    start = time.time()
    answers = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(work, range(len(batches))):
            answers.update(part)
    return answers, time.time() - start


def nal_reference(judgments, knowledge: dict, judgment_confidence: float = 0.9) -> dict | None:
    """Audit only, never a score: one complete derivation per judgment, revised once, as NAL prescribes.

    Deduction (Truth_Deduction), then for the decomposition form the structural step
    (Truth_StructuralDeduction / ...Negated with STRUCTURAL_TRUTH = (1, RELIANCE 0.9)), or for
    the negation form Truth_Negation, then revision of all of them with disjoint evidence.
    It checks whether ONA's answer is this clean derivation (no stamp truncation, no extra
    derivation paths, no lost premises). ONA's own revision is sequential and caps the
    confidence at 0.99 at every step, so tiny differences are expected near the cap.
    """
    w_sum = wf_sum = 0.0
    for j in sorted(judgments):
        if j not in knowledge:
            continue
        imp = knowledge[j]
        f = imp.frequency
        c = f * 1.0 * imp.confidence * judgment_confidence      # deduction
        if imp.form == "decomposition":
            c = c * 0.9 * f                                      # structural deduction
        f_match = 1.0 - f if imp.negated else f
        w = c / (1 - c)
        w_sum += w
        wf_sum += w * f_match
    if w_sum == 0:
        return None
    f = wf_sum / w_sum
    c = min(MAX_CONFIDENCE, w_sum / (w_sum + EVIDENTIAL_HORIZON))
    return {"frequency": f, "confidence": c, "expectation": c * (f - 0.5) + 0.5}
