"""Fellegi-Sunter, Ditto, ZeroER and AnyMatch on the official DeepMatcher splits of seven external benchmarks.

The archived Table 5 cells of these four methods on Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-Scholar,
iTunes-Amazon, Walmart-Amazon and Beer came from pipelines that sampled their own test pairs
(e.g. 2,250 per dataset for Fellegi-Sunter), whereas MatchGPT, HierGAT and NARS use the official
test.csv. This driver re-runs the four methods on the official train/valid/test splits with exactly
the protocol of run_fodors_official.py, which it imports and re-targets per dataset (module
constants DATASET, OFFICIAL_TEST, EXCLUDED, ZEROER_NAME, ANYMATCH_LEAVE_OUT); no method code is copied.

Per-dataset differences from the Fodors-Zagats driver (everything else is its code path):

* Excluded attributes: ``id`` only (no LABEL_REVEALING attribute exists on these seven).
* ZeroER self-join blocking: Fodors used upstream block_fodors_zagats on the full tables. Here each
  dataset uses its upstream ZeroER blocking function where one exists (blocking_functions.py:
  block_abt_buy, block_dblp_acm, block_dblp_scholar, block_wa for the DeepMatcher Walmart-Amazon
  schema, and block_amazon_googleproducts' rule with output attributes limited to the DeepMatcher
  columns), else the closest upstream analogue (name-token overlap 1 as block_fodors_zagats /
  block_songs) on Beer_Name and Song_Name. The left-right candidate set is always the official test
  split, so no test pair can be dropped. Self-joins run on the full tables when their candidate sets
  fit the host (``--self-join full``) and otherwise on the records that occur in the official test
  pairs (``--self-join test-records``: the only records whose self-pairs the transitivity constraint
  can consult); the mode is recorded in run.json and result.json.
* AnyMatch target: the official test pairs in AnyMatch's own column layout, obtained with the official
  preprocess.ipynb magellan_rename_columns / magellan_drop_columns (as prepare_anymatch_transfer.py),
  checked text-for-text against AnyMatch's serialization of its prepared test file. Leave-one-out
  removes the target domain from training rows, attribute pairs and external validation.
* AnyMatch DBLP rule (owner decision): DBLP-ACM and DBLP-Scholar share the DBLP table, so standard
  leave-one-out leaks target test titles through the sibling domain (748 DBLP-ACM / 964 DBLP-Scholar
  test titles in the frozen standard-LOO training corpora). For either DBLP target both DBLP domains
  (dbac, dbgo) are removed from training and validation; every other target uses standard leave-one-out.
  Each freeze counts the target's test titles found in its own external corpora (leave_one_out_check).

Outputs: experiments/completion/official-baselines/<method>/<dataset>/ and
experiments/completion/official_baselines.csv (summarize).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also under python -I (ZeroER environment)
import run_fodors_official as fo  # noqa: E402

BASE = fo.BASE
COMPLETION = fo.COMPLETION
OUT = COMPLETION / "official-baselines"
CSV = COMPLETION / "official_baselines.csv"
SEED = fo.SEED
# Official DeepMatcher test.csv (pairs, positives); cross-checked against matchgpt_external.csv and
# hiergat_external.csv by expected_test().
OFFICIAL_TEST = {"abt_buy": (1916, 206), "amazon_google": (2293, 234), "dblp_acm": (2473, 444),
                 "dblp_scholar": (5742, 1070), "itunes_amazon": (109, 27), "walmart_amazon": (2049, 193),
                 "beer": (91, 14)}
DATASETS = list(OFFICIAL_TEST)
ANYMATCH_CODE = {"abt_buy": "abt", "amazon_google": "amgo", "dblp_acm": "dbac", "dblp_scholar": "dbgo",
                 "itunes_amazon": "itam", "walmart_amazon": "waam", "beer": "beer"}
METHOD_FOLDERS = {"fellegi_sunter": "fellegi_sunter", "ditto": "ditto", "zeroer": "zeroer", "anymatch": "anymatch"}
DBLP = ["dblp_acm", "dblp_scholar"]  # share the DBLP table: the AnyMatch DBLP rule removes both for either target
DBLP_STANDARD_LOO_LEAK = {"dblp_acm": 748, "dblp_scholar": 964}  # test titles in the standard-LOO training corpus


def expected_test(dataset):
    """Official test counts, checked against the MatchGPT and HierGAT official-split results."""
    import pandas as pd
    matchgpt = pd.read_csv(COMPLETION / "matchgpt_external.csv")
    hiergat = pd.read_csv(COMPLETION / "hiergat_external.csv")
    rows = matchgpt[matchgpt.dataset.eq(dataset)]
    counts = {(int(r.n_pairs), int(r.tp + r.fn)) for r in rows.itertuples()}
    counts |= {(int(n), OFFICIAL_TEST[dataset][1]) for n in hiergat.loc[hiergat.dataset.eq(dataset), "n_test_pairs"]}
    if counts != {OFFICIAL_TEST[dataset]}:
        raise ValueError(f"{dataset}: official test counts {OFFICIAL_TEST[dataset]} disagree with MatchGPT/HierGAT {counts}")
    return OFFICIAL_TEST[dataset]


class OfficialSplit(fo.OfficialSplit):
    """run_fodors_official.OfficialSplit with the configured dataset as default name."""

    def __init__(self, name=None, data_dir=None):
        super().__init__(fo.DATASET if name is None else name, data_dir)


def configure(dataset, check_counts=True):
    """Re-target run_fodors_official to one dataset (its functions read these module globals)."""
    if dataset not in OFFICIAL_TEST:
        raise ValueError(f"Unknown dataset {dataset}; expected one of {DATASETS}")
    excluded = ["id"]
    if check_counts:  # needs the project environment (pandas, run_matchgpt_external)
        from run_matchgpt_external import LABEL_REVEALING
        excluded += LABEL_REVEALING.get(dataset, [])
        expected_test(dataset)
    fo.DATASET = dataset
    fo.OFFICIAL_TEST = OFFICIAL_TEST[dataset]
    fo.EXCLUDED = excluded
    fo.ZEROER_NAME = f"{dataset}_official"
    fo.ANYMATCH_LEAVE_OUT = ANYMATCH_CODE[dataset]
    fo.OfficialSplit = OfficialSplit
    fo.anymatch_target = anymatch_target


def stamp(output: Path, start: float, extra=None):
    """Add the driver hash, dataset and wall time to a finished result.json."""
    path = output / "result.json"
    result = json.loads(path.read_text())
    if result["dataset"] != fo.DATASET:
        raise RuntimeError(f"{path} is for {result['dataset']}, not {fo.DATASET}")
    result.update({"driver": "run_official_baselines.py", "driver_sha256": fo.sha256(Path(__file__)),
                   "wall_minutes": round((time.time() - start) / 60, 3), **(extra or {})})
    fo.dump_json(path, result)


def output_dir(method, dataset, tag=None):
    return OUT / (METHOD_FOLDERS[method] + (f"-{tag}" if tag else "")) / dataset


# ---------------------------------------------------------------- Fellegi-Sunter and Ditto
def fellegi_sunter(dataset):
    start = time.time()
    output = output_dir("fellegi_sunter", dataset)
    fo.run_fellegi_sunter(output)
    stamp(output, start)


def accumulating_train_step(original, steps):
    """ditto_light.ditto.train_step, unchanged, with each loader batch of 32 split into ``steps`` micro-batches.

    The loader (batch 32, same shuffling) is untouched. The wrapped optimizer adds each micro-batch's mean-loss
    gradient times its share n_i/N of the loader batch, and AdamW and the linear schedule step once per loader
    batch, so every update is the mean-loss gradient over the same <=32 samples as an unaccumulated run (up to
    floating-point summation order; dropout masks are drawn per micro-batch)."""
    import math

    def train_step(train_iter, model, optimizer, scheduler, hp):
        state, stash = {"weight": 1.0, "last": True}, {}
        parameters = [p for group in optimizer.param_groups for p in group["params"]]

        class Optimizer:
            def zero_grad(self, *args, **kwargs):
                optimizer.zero_grad(*args, **kwargs)

            def step(self):
                for p in parameters:
                    if p.grad is not None:
                        g = p.grad.detach() * state["weight"]
                        stash[p] = stash[p] + g if p in stash else g
                if state["last"]:
                    for p in parameters:
                        p.grad = stash.pop(p, None)
                    optimizer.step()

        class Scheduler:
            def step(self):
                if state["last"]:
                    scheduler.step()

        def micro_batches():
            for x, y in train_iter:
                size = math.ceil(len(y) / steps)
                chunks = [(x[i:i + size], y[i:i + size]) for i in range(0, len(y), size)]
                for j, (xs, ys) in enumerate(chunks):
                    state["weight"], state["last"] = len(ys) / len(y), j == len(chunks) - 1
                    yield xs, ys
        original(micro_batches(), model, Optimizer(), Scheduler(), hp)
        if stash:
            raise RuntimeError("Unapplied accumulated gradients at the end of an epoch")
    return train_step


def ditto(dataset, work, seed, gpu_gib, accumulation=1, eval_batch=None, checkpointing=False):
    """run_fodors_official.run_ditto; optionally with gradient accumulation and a cap on inference batches.

    Inference is deterministic per pair (attention masks exclude padding; run_ditto's replay check compares two
    batch sizes), so ``eval_batch`` caps the validation loader (batch 512) and the score() calls (512 and the
    replay's 17) only to bound GPU memory. ``checkpointing`` enables non-reentrant activation checkpointing
    in the RoBERTa encoder (recomputation with the saved dropout RNG state: same arithmetic, less memory)."""
    start = time.time()
    output = output_dir("ditto", dataset)
    sys.path.insert(0, str(BASE / "vendor/neural"))
    import torch.utils.data as data_module
    import ditto_light.ditto as upstream
    import run_hiergat_external as hier
    originals = (upstream.train_step, data_module.DataLoader, hier.score, upstream.DittoModel)
    extra = {"gpu": __import__("torch").cuda.get_device_name(0)}
    if accumulation > 1:
        upstream.train_step = accumulating_train_step(originals[0], accumulation)
        extra["gradient_accumulation"] = {"loader_batch": fo.DITTO["batch_size"], "micro_batches_per_update": accumulation,
                                          "micro_batch": -(-fo.DITTO["batch_size"] // accumulation),
                                          "reason": "shared 10 GB RTX 3080; effective batch 32 unchanged"}
    if eval_batch:
        class CappedLoader(originals[1]):
            def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
                super().__init__(dataset, batch_size=batch_size if shuffle else min(batch_size, eval_batch),
                                 shuffle=shuffle, **kwargs)

        def score(model, dataset, kind, batch_size=32):
            capped = min(batch_size, eval_batch)
            extra.setdefault("score_batch_sizes", []).append([batch_size, capped])
            return originals[2](model, dataset, kind, batch_size=capped)
        data_module.DataLoader, hier.score = CappedLoader, score
        extra["inference_batch_cap"] = eval_batch
    if checkpointing:
        class CheckpointedDitto(originals[3]):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        upstream.DittoModel = CheckpointedDitto
        extra["gradient_checkpointing"] = "non-reentrant, RoBERTa encoder layers"
    try:
        fo.run_ditto(output, work / dataset / "ditto", seed, gpu_gib)
    finally:
        upstream.train_step, data_module.DataLoader, hier.score, upstream.DittoModel = originals
    stamp(output, start, extra)


# ---------------------------------------------------------------- ZeroER
def block_amazon_google(A, B):
    """Upstream block_amazon_googleproducts rule (title, word level, overlap 1, no missing values); its
    output attributes are limited to the DeepMatcher columns (upstream also lists description, absent here).
    Output attributes do not change the candidate pairs: features are extracted from the tables."""
    import py_entitymatching as em
    attributes = [c for c in ["title", "description", "manufacturer", "price"] if c in A.columns and c in B.columns]
    return em.OverlapBlocker().block_tables(A, B, "title", "title", word_level=True, overlap_size=1,
                                            l_output_attrs=attributes, r_output_attrs=attributes,
                                            show_progress=False, allow_missing=False)


def name_overlap(attribute):
    """No upstream ZeroER blocking for this dataset: block_fodors_zagats / block_songs rule (token overlap 1)
    on the record name."""
    def block(A, B):
        import py_entitymatching as em
        attributes = [c for c in A.columns if c != "id"]
        return em.OverlapBlocker().block_tables(A, B, attribute, attribute, word_level=True, overlap_size=1,
                                                l_output_attrs=attributes, r_output_attrs=attributes,
                                                show_progress=False, allow_missing=False)
    block.__name__ = f"overlap1_{attribute}"
    return block


def self_join_function(dataset):
    """(function, provenance) for the self-join candidates of one dataset."""
    import blocking_functions as upstream
    named = {"abt_buy": "block_abt_buy", "dblp_acm": "block_dblp_acm", "dblp_scholar": "block_dblp_scholar",
             "walmart_amazon": "block_wa"}
    if dataset in named:
        return getattr(upstream, named[dataset]), f"upstream blocking_functions.{named[dataset]}"
    if dataset == "amazon_google":
        return block_amazon_google, ("upstream block_amazon_googleproducts rule (title overlap 1), "
                                     "output attributes limited to the DeepMatcher columns")
    attribute = {"beer": "Beer_Name", "itunes_amazon": "Song_Name"}[dataset]
    return name_overlap(attribute), (f"no upstream function: OverlapBlocker on {attribute}, word level, overlap 1 "
                                     "(block_fodors_zagats / block_songs rule)")


def zeroer_run(dataset, work: Path, jobs: int, mode: str):
    """ZeroER environment side: run_fodors_official.zeroer_run with this dataset's self-join blocking."""
    import pandas as pd
    sys.path.insert(0, str(BASE / "vendor/zeroer"))
    import py_entitymatching as em
    import blocking_functions
    function, provenance = self_join_function(dataset)
    official = pd.read_csv(work / "official_test_pairs.csv", dtype=str)
    log = []

    def self_join(A, B):
        if A is not B:
            raise RuntimeError("Self-join blocking called on two different tables")
        side = "rtable" if A.id.astype(int).min() >= fo.RIGHT_OFFSET else "ltable"
        table = A
        if mode == "test-records":
            keep = set(official[f"{side}_id"])
            table = A[A.id.astype(str).isin(keep)].reset_index(drop=True)
            em.set_key(table, "id")
        start = time.time()
        C = function(table, table)
        log.append({"side": side, "table_records": len(A), "blocked_records": len(table),
                    "candidates": len(C), "identity_pairs": int((C.ltable_id.astype(str) == C.rtable_id.astype(str)).sum()),
                    "seconds": round(time.time() - start, 1)})
        print(json.dumps({"self_join": log[-1]}), flush=True)
        return C
    # run_fodors_official.zeroer_run calls blocking_functions.block_fodors_zagats for its self-joins
    # (load_data's blocking_fn(A, A) and (B, B)); point that name at this dataset's blocking for the run.
    original = blocking_functions.block_fodors_zagats
    blocking_functions.block_fodors_zagats = self_join
    start = time.time()
    try:
        fo.zeroer_run(work, jobs)
    finally:
        blocking_functions.block_fodors_zagats = original
    run = json.loads((work / "run.json").read_text())
    run.update({"self_join_mode": mode, "self_join_blocking": provenance, "self_join_candidates": log,
                "zeroer_minutes": round((time.time() - start) / 60, 2), "driver_sha256": fo.sha256(Path(__file__))})
    fo.dump_json(work / "run.json", run)


def zeroer_collect(dataset, work: Path, tag=None):
    start = time.time()
    output = output_dir("zeroer", dataset, tag)
    run = json.loads((work / "run.json").read_text())
    if run.get("driver_sha256") is None or "self_join_mode" not in run:
        raise RuntimeError("run.json was not written by run_official_baselines.zeroer_run")
    fo.zeroer_collect(output, work)
    stamp(output, start, {"self_join_mode": run["self_join_mode"], "self_join_blocking": run["self_join_blocking"],
                          "self_join_candidates": run["self_join_candidates"], "zeroer_minutes": run["zeroer_minutes"],
                          "dropped_test_pairs": 0,
                          "left_right_candidates": "the official test pairs themselves (none dropped by blocking)"})


# ---------------------------------------------------------------- AnyMatch
def anymatch_columns(tables):
    """Official preprocess.ipynb rename/drop for the target domain (as prepare_anymatch_transfer.prepare)."""
    from prepare_anymatch_transfer import notebook_scope
    scope = notebook_scope()
    code = fo.ANYMATCH_LEAVE_OUT
    names, drop = scope["magellan_rename_columns"][code], scope["magellan_drop_columns"][code]
    columns = list(tables["tableA"].columns)
    if names[0] != "id" or len(names) != len(columns) + 1:
        raise ValueError(f"{code}: AnyMatch column names {names} do not align with the official columns {columns}")
    return dict(zip(columns, names[1:])), list(drop)


def anymatch_target(api, tokenizer):
    """Official test pairs in AnyMatch's prepared column layout, serialized by its own df_serializer(mode1)."""
    import tempfile
    import pandas as pd
    tables, pairs, source = fo.official_split()
    prepared = api.data_root("feature-clean") / "prepared" / fo.ANYMATCH_LEAVE_OUT
    rename, drop = anymatch_columns(tables)
    a, b = (tables[s].rename(columns=rename).drop(columns=drop) for s in ["tableA", "tableB"])
    columns = list(a.columns)
    test = pairs["test"].reset_index(drop=True)
    frame = pd.concat([a.loc[test.ltable_id].reset_index(drop=True).add_suffix("_l"), test[["label"]],
                       b.loc[test.rtable_id].reset_index(drop=True).add_suffix("_r")], axis=1)
    reference = pd.read_csv(prepared / "test.csv")
    if list(frame.columns) != list(reference.columns):
        raise ValueError(f"Column layout {list(frame.columns)} differs from AnyMatch's prepared {list(reference.columns)}")
    staged = Path(tempfile.mkdtemp()) / "test.csv"
    frame.to_csv(staged, index=False)  # same CSV round trip (pandas NA parsing) as read_single_row_data
    module = api.official_data()
    serialized = module.df_serializer(pd.read_csv(staged), "mode1")
    official = module.df_serializer(reference.copy(), "mode1")
    staged.unlink(); staged.parent.rmdir()
    # AnyMatch's preparation merge reorders the pairs; the serialized (text, label) multisets must be identical.
    if sorted(zip(serialized.text, serialized.label)) != sorted(zip(official.text, official.label)):
        raise ValueError(f"Official-split serialization differs from AnyMatch's prepared {fo.ANYMATCH_LEAVE_OUT} test serialization")
    if set(fo.EXCLUDED) & set(columns):
        raise ValueError("Excluded attribute in the AnyMatch target")
    return serialized, test, source, columns


def anymatch_left_out(dataset):
    """AnyMatch source domains removed for this target: the target, and for a DBLP target both DBLP domains."""
    return [ANYMATCH_CODE[d] for d in DBLP] if dataset in DBLP else [ANYMATCH_CODE[dataset]]


def anymatch_rule_note(dataset):
    if dataset not in DBLP:
        return "standard leave-one-out: only the target domain is removed from AnyMatch training and validation"
    sibling = next(d for d in DBLP if d != dataset)
    return (f"DBLP rule (owner decision): {dataset} shares the DBLP table with {sibling}, so standard leave-one-out "
            f"leaks {DBLP_STANDARD_LOO_LEAK[dataset]} {dataset} test titles into training; both DBLP domains "
            f"({', '.join(anymatch_left_out(dataset))}) are excluded from AnyMatch training and validation")


def target_title_overlap(tables, pairs, frames):
    """Target test records (first attribute, >= 12 characters, as AnyMatch serializes it) found verbatim in each
    external corpus; the same count as the frozen leave-one-out check that measured the DBLP leak."""
    first = tables["tableA"].columns[0]
    test = pairs["test"]
    values = sorted({str(v).strip().lower() for side, col in [("tableA", "ltable_id"), ("tableB", "rtable_id")]
                     for v in tables[side].loc[test[col], first] if len(str(v).strip()) >= 12})
    counts = {"field": first, "distinct_test_values": len(values)}
    for name, frame in frames.items():
        text = "\n".join(frame.text.str.lower())
        counts[name] = sum(1 for v in values if f"col {v}," in text or f"col {v}</p>" in text)
    return counts


def anymatch_freeze(dataset, work: Path, checkpointing=False):
    """run_fodors_official.anymatch_freeze for this target, with the protocol's target description corrected.

    ``checkpointing`` sets the protocol's gradient_checkpointing (anymatch_training enables HF GPT-2 activation
    checkpointing; recomputation restores the dropout RNG state, so the updates are the same arithmetic) to fit
    a ~4 GiB share of the desktop RTX 3080; recorded as a resource deviation."""
    protocol_dir = work / dataset / "anymatch/protocol"
    api, _ = fo.anymatch_api()
    left_out = anymatch_left_out(dataset)
    original = api.MAGELLAN
    if not set(left_out) <= set(original):
        raise ValueError(f"{left_out} are not all AnyMatch source domains {list(original)}")
    # fo.anymatch_freeze removes the target itself (official loo.py get_loo_dirs); the DBLP sibling goes here.
    api.MAGELLAN = {k: v for k, v in original.items() if k not in left_out or k == fo.ANYMATCH_LEAVE_OUT}
    try:
        fo.anymatch_freeze(protocol_dir)
    finally:
        api.MAGELLAN = original
    sources = [k for k in list(original) + ["wdc"] if k not in left_out]
    import pandas as pd  # noqa: F401  (after run_anymatch_transfer, imported by anymatch_api)
    tables, pairs, _ = fo.official_split()
    rename, drop = anymatch_columns(tables)
    path = protocol_dir / "protocol.json"
    protocol = json.loads(path.read_text())
    train = pd.read_csv(protocol_dir / "external_train.csv")
    valid = pd.read_csv(protocol_dir / "external_valid.csv")
    protocol.update(
        target=(f"{dataset} official DeepMatcher test split ({len(pairs['test'])} pairs), id excluded; "
                f"AnyMatch preprocessing drops {drop or 'nothing'}"),
        target_fields=[rename[c] for c in tables["tableA"].columns if rename[c] not in drop])
    domain = {"abt": "Abt-Buy", "amgo": "Amazon-Google", "dbac": "DBLP-ACM", "dbgo": "DBLP-Scholar",
              "itam": "iTunes-Amazon", "waam": "Walmart-Amazon", "beer": "Beer"}[fo.ANYMATCH_LEAVE_OUT]
    protocol["deviations"] = [d.replace("Leave-one-out target Fodors-Zagats", f"Leave-one-out target {domain}")
                              for d in protocol["deviations"]]
    if not any(domain in d for d in protocol["deviations"]):
        raise RuntimeError("Leave-one-out deviation text not found in the frozen protocol")
    if dataset in DBLP:
        eight = "the other eight source domains"
        if sum(eight in d for d in protocol["deviations"]) != 1:
            raise RuntimeError("Source-domain count text not found in the frozen protocol")
        protocol["deviations"] = [d.replace(eight, "the other seven source domains (DBLP rule)") for d in protocol["deviations"]]
        protocol["deviations"].append(anymatch_rule_note(dataset) + " (rows, attribute pairs, AutoML receipts).")
        protocol["automl_receipts"] = {k: v for k, v in protocol["automl_receipts"].items()
                                       if not any(f"/{code}/" in k for code in left_out)}
    if checkpointing:
        protocol["gradient_checkpointing"] = True
        protocol["deviations"].append("Resource only: gradient checkpointing on (activation recomputation, same "
                                      "updates) to fit a ~4 GiB share of a 10 GiB RTX 3080; batch, data and optimizer unchanged.")
    overlap = target_title_overlap(tables, pairs, {"external_train": train, "external_valid": valid})
    protocol["leave_one_out_check"] = {"external_train_rows": len(train), "external_valid_rows": len(valid),
                                       "left_out_domains": left_out, "source_domains": sources,
                                       "rule": anymatch_rule_note(dataset), "target_test_titles_in_corpus": overlap,
                                       "driver_sha256": fo.sha256(Path(__file__))}
    fo.dump_json(path, protocol)
    print(json.dumps({"target": protocol["target"], "target_fields": protocol["target_fields"],
                      "leave_one_out_check": protocol["leave_one_out_check"]}))


ANYMATCH_RAM_GATE_GIB = 3.0


def anymatch_train(dataset, work: Path, seed: int, gpu_gib: float):
    """run_fodors_official.anymatch_train with resource gates adapted to the host; training code and protocol untouched.

    Host RAM gate 3 GiB instead of the helper's 8 GiB (the job runs inside a systemd MemoryMax cgroup, which
    bounds it harder). On a GPU below 15 GiB the GPU gates are adapted as described below and recorded in
    <model_dir>/resource_gate_adaptation.json."""
    api, training = fo.anymatch_api()
    import resource_limits

    def check_memory(minimum_gib=8):
        return resource_limits.check_memory(min(minimum_gib, ANYMATCH_RAM_GATE_GIB))
    training.check_memory = check_memory
    original = fo.guard_resources
    fo.guard_resources = lambda min_ram_gib=3.0: original(min(min_ram_gib, ANYMATCH_RAM_GATE_GIB))
    import torch
    cuda = torch.cuda
    real_fraction, real_info = cuda.set_per_process_memory_fraction, cuda.mem_get_info
    gpu_log = {"device": cuda.get_device_name(0), "total_gib": cuda.get_device_properties(0).total_memory / 1024**3}
    small_gpu = gpu_log["total_gib"] < 15
    if small_gpu:
        # The helper's gates assume the 16 GiB laptop GPU: >= 9 GiB free and a 0.5 per-process cap. On the shared
        # 10 GiB desktop RTX 3080 9 GiB is never free; the cap becomes gpu_gib and the free-memory gate gpu_gib.
        # Same model, batch and numerics; only the gates change.
        cap = gpu_gib * 1024**3 / cuda.get_device_properties(0).total_memory

        def fraction(value, *args):
            gpu_log.update(requested_fraction=value, applied_fraction=cap, cap_gib=gpu_gib)
            return real_fraction(cap, *args)

        def mem_get_info(*args):
            free, total = real_info(*args)
            gpu_log.setdefault("free_gib_at_gate", free / 1024**3)
            if free < gpu_gib * 1024**3:
                raise RuntimeError(f"Insufficient GPU headroom: {free / 1024**3:.1f} GiB free, {gpu_gib} GiB needed")
            return max(free, 9 * 1024**3), total  # passes the helper's 9 GiB gate once our gpu_gib gate holds
        cuda.set_per_process_memory_fraction, cuda.mem_get_info = fraction, mem_get_info
    model_dir = work / dataset / f"anymatch/seed_{seed}"
    try:
        fo.anymatch_train(work / dataset / "anymatch/protocol", model_dir, seed)
    finally:
        fo.guard_resources = original
        cuda.set_per_process_memory_fraction, cuda.mem_get_info = real_fraction, real_info
    gpu_log["gates_adapted"] = small_gpu
    fo.dump_json(model_dir / "resource_gate_adaptation.json", {**gpu_log, "host_ram_gate_gib": ANYMATCH_RAM_GATE_GIB,
                                                               "driver_sha256": fo.sha256(Path(__file__))})


def anymatch_score(dataset, work: Path, seed: int):
    start = time.time()
    output = output_dir("anymatch", dataset)
    model_dir = work / dataset / f"anymatch/seed_{seed}"
    protocol_dir = work / dataset / "anymatch/protocol"
    fo.anymatch_score(output, protocol_dir, model_dir)
    check = json.loads((protocol_dir / "protocol.json").read_text())["leave_one_out_check"]
    if check["left_out_domains"] != anymatch_left_out(dataset):
        raise RuntimeError(f"Protocol left out {check['left_out_domains']}, the rule requires {anymatch_left_out(dataset)}")
    result = json.loads((output / "result.json").read_text())
    words = {7: "seven", 8: "eight"}
    result["protocol"]["transfer"] = (f"leave-one-out: trained on the {words[len(check['source_domains'])]} other AnyMatch "
                                      f"source domains ({', '.join(check['source_domains'])})")
    result["protocol"]["serializer"] = result["protocol"]["serializer"].replace("Fodors-Zagats", dataset)
    result["notes"] = [check["rule"], f"target test titles found in the external corpora: "
                       f"{json.dumps(check['target_test_titles_in_corpus'])}"]
    fo.dump_json(output / "result.json", result)
    adaptation = json.loads((model_dir / "resource_gate_adaptation.json").read_text())
    fo.dump_json(output / "source_fit/resource_gate_adaptation.json", adaptation)
    stamp(output, start, {"resource_gate_adaptation": adaptation})


# ---------------------------------------------------------------- summary
PROTOCOL_NOTES = {
    "fellegi_sunter": "Smoothed m/u agreement model over every attribute except id; fit on train; validation-F1 threshold (run_baseline_eval evaluator)",
    "ditto": fo.PROTOCOL_NOTES["ditto"],
    "zeroer": "Official ZeroER, Magellan similarity features, two-table mode with transitivity; L-R candidates = official test pairs; unsupervised, posterior 0.5",
    "anymatch": "AnyMatch GPT-2 leave-one-out transfer from the other source domains (eight; seven for DBLP targets: both DBLP domains removed); external-validation checkpoint; fixed 0.5; one source fit",
}


def summarize():
    import pandas as pd
    rows = []
    for method, folder in METHOD_FOLDERS.items():
        for dataset in DATASETS:
            path = OUT / folder / dataset / "result.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            frame = pd.read_csv(path.parent / "test_predictions.csv")
            again = fo.recomputed(frame)
            if any(abs(again[k] - r["test"][k]) > 1e-12 for k in ["precision", "recall", "f1"]) or \
                    fo.sha256(path.parent / "test_predictions.csv") != r["test_predictions_sha256"]:
                raise RuntimeError(f"{path}: saved predictions do not reproduce the result")
            if (r["method"], r["dataset"]) != (method, dataset) or not r["identical_to_official_test_csv"] or \
                    (r["n_test_pairs"], r["n_positive"]) != OFFICIAL_TEST[dataset]:
                raise RuntimeError(f"{path}: not the official {dataset} test split")
            notes = [f"{r.get('wall_minutes', '?')} min wall"]
            if method == "ditto":
                notes.append(f"best epoch {r['best_epoch']}, valid F1 {r['valid_f1']:.4f}, truncated test rows {r['truncated_rows']['test']}")
            if method == "zeroer":
                sizes = "/".join(str(c["candidates"]) for c in r["self_join_candidates"])
                notes.append(f"self-joins {r['self_join_mode']} ({sizes} L-L/R-R candidates; {r['self_join_blocking']}); 0 test pairs dropped")
            if method == "anymatch":
                notes.append(f"checkpoint epoch {r['best_epoch']} of {r['epochs_run']}, external validation F1 {r['best_external_validation_f1']:.4f}")
                notes.extend(r.get("notes", []))
            seeds = r["seeds"]
            rows.append({"method": method, "dataset": dataset,
                         "seeds": ";".join(map(str, seeds)) if isinstance(seeds, list) else "deterministic",
                         "n_test_pairs": r["n_test_pairs"], "n_positive": r["n_positive"], "threshold": r["threshold"],
                         "precision": again["precision"], "recall": again["recall"], "f1": again["f1"], "sd": "",
                         "tp": again["tp"], "fp": again["fp"], "fn": again["fn"],
                         "excluded_attributes": ";".join(r["excluded_input_attributes"]),
                         "protocol": PROTOCOL_NOTES[method], "result": str(path.relative_to(COMPLETION)),
                         "notes": "; ".join(notes)})
    frame = pd.DataFrame(rows)
    frame.to_csv(CSV, index=False, float_format="%.6f")
    print(frame.drop(columns=["protocol", "notes", "result"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fellegi-sunter", "ditto", "zeroer-export", "zeroer-run", "zeroer-collect",
                                            "anymatch-freeze", "anymatch-train", "anymatch-score", "summarize"])
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--work", type=Path, help="scratch directory outside the tree (checkpoints, serialized inputs)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu-memory-gib", type=float, default=12.0)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Ditto encoder / AnyMatch protocol activation checkpointing (GPU memory only)")
    parser.add_argument("--eval-batch", type=int, help="Ditto inference batch cap (GPU memory only)")
    parser.add_argument("--accumulation", type=int, default=1, help="Ditto micro-batches per update (effective batch stays 32)")
    parser.add_argument("--self-join", choices=["full", "test-records"], default="test-records")
    parser.add_argument("--tag", help="ZeroER sensitivity output folder suffix (zeroer-<tag>)")
    args = parser.parse_args()
    if args.command.startswith("anymatch"):
        import run_anymatch_transfer  # noqa: F401  (before pandas: dependency overlay)
    if args.command == "summarize":
        summarize()
        sys.exit()
    if args.dataset is None:
        parser.error("--dataset is required")
    configure(args.dataset, check_counts=args.command != "zeroer-run")
    zeroer_work = (args.work / args.dataset / ("zeroer" + (f"-{args.tag}" if args.tag else ""))) if args.work else None
    if args.command == "fellegi-sunter":
        fellegi_sunter(args.dataset)
    elif args.command == "ditto":
        ditto(args.dataset, args.work, args.seed, args.gpu_memory_gib, args.accumulation, args.eval_batch,
              args.gradient_checkpointing)
    elif args.command == "zeroer-export":
        fo.zeroer_export(zeroer_work)
    elif args.command == "zeroer-run":
        zeroer_run(args.dataset, zeroer_work, args.jobs, args.self_join)
    elif args.command == "zeroer-collect":
        zeroer_collect(args.dataset, zeroer_work, args.tag)
    elif args.command == "anymatch-freeze":
        anymatch_freeze(args.dataset, args.work, args.gradient_checkpointing)
    elif args.command == "anymatch-train":
        anymatch_train(args.dataset, args.work, args.seed, args.gpu_memory_gib)
    elif args.command == "anymatch-score":
        anymatch_score(args.dataset, args.work, args.seed)
