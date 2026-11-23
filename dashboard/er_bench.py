from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .eval_api import _load_gt, analyze_backends, _sha256_file


@dataclass
class BenchResult:
    """Container for a single model's benchmark result.

    Args:
        name (str): Model name.
        status (str): Run status string.
        metric (Optional[str]): Metric string such as F1=0.87.
        details (Optional[Dict]): Optional structured details.

    Returns:
        BenchResult: A result record.
    """
    name: str
    status: str
    metric: Optional[str] = None
    details: Optional[Dict] = None


def _record_text(row: pd.Series) -> str:
    """Builds a single text field from key attributes.

    Args:
        row (pd.Series): Row with name and birth attributes.

    Returns:
        str: Concatenated descriptive text.
    """
    parts: List[str] = []
    for c in [
        "nafn_norm",
        "first_name",
        "middle_name",
        "patronym",
        "surname",
        "sex",
        "status",
        "marriagestatus",
    ]:
        v = str(row.get(c, "") or "").strip()
        if v:
            parts.append(v)
    by = row.get("birthyear", None)
    if by is not None and not pd.isna(by) and str(by) != "":
        parts.append(f"by:{int(by)}")
    parish = row.get("parish", None)
    if parish is not None and not pd.isna(parish) and str(parish) != "":
        parts.append(f"parish:{int(parish)}")
    return " ".join(parts) if parts else ""


def _pairs_from_clusters(
    people: pd.DataFrame,
    gt: pd.DataFrame,
    max_pos: int = 2000,
    neg_ratio: float = 1.0,
    seed: int = 42,
    adjacent_waves: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """Creates labeled pairs for EM from cluster ground truth.

    Args:
        people (pd.DataFrame): Canonical people dataframe.
        gt (pd.DataFrame): DataFrame with columns id and gt_cluster.
        max_pos (int): Maximum number of positive pairs to sample.
        neg_ratio (float): Negatives per positive.
        seed (int): Random seed.

    Returns:
        pd.DataFrame: Pairs with left_id, right_id, left_text, right_text, label.
    """
    rng = random.Random(seed)
    people_idx = people.set_index("id")
    id_to_text = people.apply(_record_text, axis=1)
    id_to_text.index = people["id"].values
    cl_map: Dict[int, int] = {}
    for rid, cid in gt.itertuples(index=False):
        if pd.isna(cid) or int(rid) not in people_idx.index:
            continue
        cl_map[int(rid)] = int(cid)
    cluster_to_ids: Dict[int, List[int]] = {}
    for rid, cid in cl_map.items():
        cluster_to_ids.setdefault(cid, []).append(rid)
    pos_pairs: List[Tuple[int, int]] = []
    for members in cluster_to_ids.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if adjacent_waves is not None:
                    wa, wb = int(people_idx.at[a, "heimild"]), int(people_idx.at[b, "heimild"])
                    if wa > wb:
                        a, b, wa, wb = b, a, wb, wa
                    if adjacent_waves.get(wa) != wb:
                        continue
                pos_pairs.append((a, b))
    rng.shuffle(pos_pairs)
    pos_pairs = pos_pairs[: max_pos]
    # Unknown identities are never negatives. A failed hard-negative search
    # must not fall back to an arbitrary (possibly positive or self) record.
    pool_ids = sorted(cl_map)
    wave_pools, prefix_pools = {}, {}
    for rid in pool_ids:
        wave = int(people_idx.at[rid, "heimild"]) if adjacent_waves is not None else None
        wave_pools.setdefault(wave, []).append(rid)
        prefix_pools.setdefault((wave, id_to_text.get(rid, "")[:2]), []).append(rid)
    used = {tuple(sorted(p)) for p in pos_pairs}
    def plausible_neg(a: int, target: int) -> Optional[int]:
        ta = id_to_text.get(a, "")
        by = people_idx.loc[a, "birthyear"] if a in people_idx.index else None
        wave = int(people_idx.at[target, "heimild"]) if adjacent_waves is not None else None
        def available(pool):
            return [b for b in pool if b != a and cl_map[b] != cl_map[a]
                    and tuple(sorted((a, b))) not in used]
        hard_pool = prefix_pools.get((wave, ta[:2]), [])
        eligible = available(rng.sample(hard_pool, min(50, len(hard_pool))))
        if not eligible:
            pool = wave_pools.get(wave, [])
            eligible = available(rng.sample(pool, min(100, len(pool))))
            if not eligible:
                eligible = available(pool)
        for b in eligible[:50]:
            tb = id_to_text.get(b, "")
            ok_year = True
            if by is not None and not pd.isna(by) and str(by) != "":
                try:
                    byb = int(people_idx.loc[b, "birthyear"]) if b in people_idx.index else None
                    ok_year = byb is None or pd.isna(byb) or abs(int(by) - int(byb)) <= 5
                except Exception:
                    ok_year = True
            ok_prefix = True
            if ta and tb:
                ok_prefix = ta[:2] == tb[:2]
            if ok_year and ok_prefix:
                return b
        return eligible[0] if eligible else None
    neg_pairs: List[Tuple[int, int]] = []
    if neg_ratio < 0:
        raise ValueError("Negative sampling ratio must be nonnegative")
    target_negatives = int(round(len(pos_pairs) * neg_ratio))
    for i in range(target_negatives):
        a, target = pos_pairs[i % len(pos_pairs)]
        b = plausible_neg(a, target)
        if b is None:
            raise ValueError("Insufficient distinct labelled negatives; reduce requested pair budget")
        used.add(tuple(sorted((a, b))))
        neg_pairs.append((a, b))
    rows: List[Dict] = []
    for a, b in pos_pairs:
        rows.append({
            "left_id": int(a),
            "right_id": int(b),
            "left_text": id_to_text.get(a, ""),
            "right_text": id_to_text.get(b, ""),
            "label": 1,
        })
    for a, b in neg_pairs:
        rows.append({
            "left_id": int(a),
            "right_id": int(b),
            "left_text": id_to_text.get(a, ""),
            "right_text": id_to_text.get(b, ""),
            "label": 0,
        })
    if not rows:
        # Return empty DataFrame with expected columns if no pairs
        pairs = pd.DataFrame(columns=["left_id", "right_id", "left_text", "right_text", "label"])
    else:
        pairs = pd.DataFrame(rows)
        pairs = pairs.dropna(subset=["left_id", "right_id"]).copy()
    return pairs


def prepare_pair_splits(
    people_csv: str,
    labels_csv: str,
    out_dir: str,
    max_pos: int = 2000,
    neg_ratio: float = 1.0,
    seed: int = 42,
    sample_frac: float = 1.0,
    sample_seed: int = 42,
    split_strategy: str = "temporal",
) -> str:
    """Prepares pair classification CSV splits from ICE-ID people and labels.

    Args:
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels csv.
        out_dir (str): Output directory to write train/validation/test CSV files.
        max_pos (int): Maximum positive pairs.
        neg_ratio (float): Negatives per positive.
        seed (int): Sampling seed.
        sample_frac (float): Fraction of people rows to sample prior to pairing.
        sample_seed (int): Seed for people sampling.

    Returns:
        str: Path to the prepared dataset directory.
    """
    parameters={'max_pos':max_pos,'neg_ratio':neg_ratio,'seed':seed,'sample_frac':sample_frac,
                'sample_seed':sample_seed,'split_strategy':split_strategy}
    input_hashes={'people_sha256':_sha256_file(people_csv),
                  'labels_sha256':_sha256_file(labels_csv) if os.path.isfile(labels_csv) else None}
    destination=Path(out_dir);manifest_path=destination/'split_manifest.json'
    if destination.exists() and any(destination.iterdir()):
        if not manifest_path.exists():raise ValueError('Preserve partial pair export; choose a new empty output directory')
        saved=json.loads(manifest_path.read_text())
        if saved.get('status')!='complete' or saved.get('parameters')!=parameters or any(saved.get(k)!=v for k,v in input_hashes.items()):
            raise ValueError('Existing pair export has different or unbound inputs; choose a new empty output directory')
        if any(not (destination/name).is_file() or _sha256_file(str(destination/name))!=digest
               for name,digest in saved.get('output_sha256',{}).items()) or len(saved.get('output_sha256',{}))!=5:
            raise ValueError('Existing pair export is incomplete or changed; preserve it and choose a new empty directory')
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    for c in [
        "id",
        "nafn_norm",
        "first_name",
        "middle_name",
        "patronym",
        "surname",
        "birthyear",
        "sex",
        "status",
        "marriagestatus",
        "parish",
    ]:
        if c not in people.columns:
            people[c] = ""
    people["id"] = pd.to_numeric(people["id"], errors="coerce").astype("Int64")
    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype("int64", copy=False)
    for c in ["birthyear", "parish"]:
        people[c] = pd.to_numeric(people[c], errors="coerce").astype("Int64")
    if 0.0 < sample_frac < 1.0:
        people = people.sample(frac=sample_frac, random_state=sample_seed).copy()
    gt = _load_gt(people_csv, labels_csv)
    gt = gt[gt["id"].isin(people["id"])].copy()
    if split_strategy not in {"temporal", "legacy_random_pairs"}:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    # Build left/right tables with a simple 'text' attribute
    tbl = people[["id"]].copy()
    tbl["text"] = people.apply(_record_text, axis=1)
    left_fp = os.path.join(out_dir, "left.csv")
    right_fp = os.path.join(out_dir, "right.csv")
    tbl.to_csv(left_fp, index=False)
    tbl.to_csv(right_fp, index=False)
    def pair_table(frame, mapping=None):
        local_gt = gt[gt.id.isin(frame.id)]
        pairs = _pairs_from_clusters(frame, local_gt, max_pos=max_pos, neg_ratio=neg_ratio,
                                     seed=seed, adjacent_waves=mapping)
        result = pairs[["left_id", "right_id", "label"]].rename(
            columns={"left_id": "ltable_id", "right_id": "rtable_id"})
        result.insert(0, "id", np.arange(1, len(result) + 1))
        return result
    if split_strategy == "temporal":
        if "heimild" not in people:
            raise ValueError("Temporal pair construction requires census year (heimild)")
        people["heimild"] = pd.to_numeric(people.heimild, errors="raise").astype(int)
        waves = sorted(people.heimild.unique())
        next_wave = dict(zip(waves, waves[1:]))
        partitions = {"train.csv": people[people.heimild.le(1870)],
                      "validation.csv": people[people.heimild.between(1871, 1890)],
                      "test.csv": people[people.heimild.ge(1891)]}
        splits = {name: pair_table(frame, next_wave) for name, frame in partitions.items()}
        if any(frame.empty for frame in splits.values()):
            raise ValueError("A temporal partition has no eligible adjacent-wave reference pairs")
    else:
        pairs_dm = pair_table(people)
        idx = np.random.default_rng(seed).permutation(len(pairs_dm))
        n_train, n_valid = int(.7 * len(idx)), int(.15 * len(idx))
        splits = {"train.csv": pairs_dm.iloc[idx[:n_train]].copy(),
                  "validation.csv": pairs_dm.iloc[idx[n_train:n_train+n_valid]].copy(),
                  "test.csv": pairs_dm.iloc[idx[n_train+n_valid:]].copy()}
    for fn, df in splits.items():
        df.to_csv(os.path.join(out_dir, fn), index=False)
    with open(os.path.join(out_dir, "split_manifest.json"), "w") as stream:
        json.dump({'status':'complete','parameters':parameters,"split_strategy": split_strategy, "people_csv": os.path.abspath(people_csv),
                   "labels_csv": os.path.abspath(labels_csv), "seed": seed,
                   **input_hashes,
                   'output_sha256':{name:_sha256_file(os.path.join(out_dir,name)) for name in [*splits,'left.csv','right.csv']},
                   "reference_warning": "Reference IDs are not independently adjudicated historical truth; unknown IDs excluded from supervision",
                   "counts": {name: {"pairs": len(frame), "reference_positive": int(frame.label.sum())}
                              for name, frame in splits.items()}}, stream, indent=2)
    return out_dir


def run_deepmatcher(dataset_dir: str, out_dir: str) -> BenchResult:
    return BenchResult(name="DeepMatcher", status="fail", metric=None, details={"error": "removed"})


def run_sklearn_baseline(dataset_dir: str, out_dir: str) -> BenchResult:
    """Runs a simple TF-IDF + Logistic Regression baseline on pairs.

    Args:
        dataset_dir (str): Directory with train/validation/test CSVs and left/right tables.
        out_dir (str): Directory to save artifacts.

    Returns:
        BenchResult: Result with status and F1 metric if available.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
    except Exception as e:
        return BenchResult(name="TFIDF_LogReg", status="fail", metric=None, details={"error": str(e)})
    try:
        os.makedirs(out_dir, exist_ok=True)
        left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
        right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
        def join_text(df):
            return pd.Series(df["id"]).to_frame().merge(left, on="id", how="left").rename(columns={"text": "lt"}).merge(right, on="id", how="left").rename(columns={"text": "rt"})
        def load_pairs(fp):
            p = pd.read_csv(fp)
            l = left.rename(columns={"id": "ltable_id"})[["ltable_id", "text"]].rename(columns={"text": "lt"})
            r = right.rename(columns={"id": "rtable_id"})[["rtable_id", "text"]].rename(columns={"text": "rt"})
            q = p.merge(l, on="ltable_id", how="left").merge(r, on="rtable_id", how="left")
            X = (q["lt"].fillna("") + " [SEP] " + q["rt"].fillna(""))
            y = q["label"].astype(int)
            return X, y
        Xtr, ytr = load_pairs(os.path.join(dataset_dir, "train.csv"))
        Xva, yva = load_pairs(os.path.join(dataset_dir, "validation.csv"))
        Xte, yte = load_pairs(os.path.join(dataset_dir, "test.csv"))
        vect = TfidfVectorizer(min_df=2, max_features=50000)
        Xtrv = vect.fit_transform(Xtr)
        Xvav = vect.transform(Xva)
        Xtev = vect.transform(Xte)
        clf = LogisticRegression(max_iter=200)
        clf.fit(Xtrv, ytr)
        ypred = clf.predict(Xtev)
        f1 = f1_score(yte, ypred)
        return BenchResult(name="TFIDF_LogReg", status="ok", metric=f"F1={f1:.4f}")
    except Exception as e:
        try:
            with open(os.path.join(out_dir, "error.txt"), "w") as f:
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        return BenchResult(name="TFIDF_LogReg", status="fail", metric=None, details={"error": str(e)})


def summarize_ours(run_dir: str, people_csv: str, labels_csv: str) -> List[BenchResult]:
    """Summarizes current pipeline backends for quick comparison.

    Args:
        run_dir (str): Run directory containing backend subfolders.
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels.csv.

    Returns:
        List[BenchResult]: One result per backend with precision metrics.
    """
    res = analyze_backends(run_dir, people_csv, labels_csv)
    out: List[BenchResult] = []
    for k, v in res.items():
        if k == "_debug":
            continue
        metric = None
        if isinstance(v, dict):
            pl = v.get("prec_labeled", None)
            if pl is not None:
                metric = f"PrecLabeled={pl}"
        out.append(BenchResult(name=str(k), status="ok", metric=metric, details=v if isinstance(v, dict) else None))
    return out


def run_all(people_csv: str, labels_csv: str, run_dir: str, out_root: str, sample_frac: float = 1.0) -> str:
    """Runs available ER models and writes a comparison table.

    Args:
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels.csv.
        run_dir (str): Path to our pipeline run directory containing backends.
        out_root (str): Output root directory to write artifacts and results.
        sample_frac (float): Fraction of people rows to sample for EM prep.

    Returns:
        str: Path to the results CSV file.
    """
    os.makedirs(out_root, exist_ok=True)
    rows: List[Dict[str, str]] = []
    details_rows: List[Dict[str, object]] = []
    dm_data = os.path.join(out_root, "pair_splits")
    prepare_pair_splits(
        people_csv,
        labels_csv,
        dm_data,
        max_pos=1000000 if sample_frac >= 1.0 else 1000,
        neg_ratio=1.0,
        seed=42,
        sample_frac=sample_frac,
        sample_seed=42,
    )
    # DeepMatcher removed

    # Add TF-IDF baseline for immediate comparison
    sk_out = os.path.join(out_root, "sklearn_baseline")
    sk_res = run_sklearn_baseline(dm_data, sk_out)
    rows.append({"model": sk_res.name, "status": sk_res.status, "metric": sk_res.metric or "-"})
    details_rows.append({"model": sk_res.name, "status": sk_res.status, "metric": sk_res.metric, "details": sk_res.details})

    # Ditto reimplementation via HF classifier on DM splits
    ditto_reimpl_out = os.path.join(out_root, "ditto_reimpl")
    ditto_reimpl_res = run_ditto_reimpl(dm_data, ditto_reimpl_out)
    rows.append({"model": ditto_reimpl_res.name, "status": ditto_reimpl_res.status, "metric": ditto_reimpl_res.metric or "-"})
    details_rows.append({"model": ditto_reimpl_res.name, "status": ditto_reimpl_res.status, "metric": ditto_reimpl_res.metric, "details": ditto_reimpl_res.details})

    # HF DistilBERT fine-tune on pairs
    hf_out = os.path.join(out_root, "hf_distilbert")
    hf_res = run_hf_classifier(dm_data, hf_out)
    rows.append({"model": hf_res.name, "status": hf_res.status, "metric": hf_res.metric or "-"})
    details_rows.append({"model": hf_res.name, "status": hf_res.status, "metric": hf_res.metric, "details": hf_res.details})

    # SBERT zero-shot style
    sbert_out = os.path.join(out_root, "sbert_zeroer")
    sbert_res = run_sbert_zeroer(dm_data, sbert_out)
    rows.append({"model": sbert_res.name, "status": sbert_res.status, "metric": sbert_res.metric or "-"})
    details_rows.append({"model": sbert_res.name, "status": sbert_res.status, "metric": sbert_res.metric, "details": sbert_res.details})
    ext = _try_external_models()
    rows.extend(ext)
    for r in summarize_ours(run_dir, people_csv, labels_csv):
        rows.append({"model": r.name, "status": r.status, "metric": r.metric or "-"})
        details_rows.append({"model": r.name, "status": r.status, "metric": r.metric, "details": r.details})
    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(out_root, "er_results.csv")
    res_df.to_csv(out_csv, index=False)
    with open(os.path.join(out_root, "er_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_root, "er_details.json"), "w") as f:
        json.dump(details_rows, f, indent=2)
    return out_csv


def prepare_ditto_data(people_csv: str, labels_csv: str, out_dir: str, max_pos: int = 2000, neg_ratio: float = 1.0, seed: int = 42, sample_frac: float = 1.0, sample_seed: int = 42) -> str:
    """Writes Ditto text triplets files train.txt/valid.txt/test.txt from people/labels.

    Args:
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels.csv.
        out_dir (str): Destination directory to write split files.
        max_pos (int): Maximum positive pairs before sampling.
        neg_ratio (float): Negatives per positive.
        seed (int): Pair sampling seed.
        sample_frac (float): Fraction of people to sample.
        sample_seed (int): Seed for people sampling.

    Returns:
        str: Path to the directory containing split files.
    """
    os.makedirs(out_dir, exist_ok=True)
    tmp_dm = os.path.join(out_dir, "_tmp_dm")
    prepare_pair_splits(people_csv, labels_csv, tmp_dm, max_pos=max_pos, neg_ratio=neg_ratio, seed=seed, sample_frac=sample_frac, sample_seed=sample_seed)
    left = pd.read_csv(os.path.join(tmp_dm, "left.csv"))
    right = pd.read_csv(os.path.join(tmp_dm, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    def dump_split(csv_name: str, out_name: str):
        df = pd.read_csv(os.path.join(tmp_dm, csv_name))
        with open(os.path.join(out_dir, out_name), "w") as f:
            for _, row in df.iterrows():
                lt = lmap.get(int(row["ltable_id"]), "")
                rt = rmap.get(int(row["rtable_id"]), "")
                lab = int(row["label"]) if not pd.isna(row["label"]) else 0
                f.write(f"{lt} ||| {rt} ||| {lab}\n")
    iceid_dir = os.path.join(out_dir, "ICEID")
    os.makedirs(iceid_dir, exist_ok=True)
    dump_split("train.csv", os.path.join("ICEID", "train.txt"))
    dump_split("validation.csv", os.path.join("ICEID", "valid.txt"))
    dump_split("test.csv", os.path.join("ICEID", "test.txt"))
    return out_dir


def run_ditto_from_people(people_csv: str, labels_csv: str, out_dir: str, sample_frac: float = 1.0) -> BenchResult:
    """Runs Ditto training for 1 epoch on prepared ICEID triplets and extracts F1.

    Args:
        people_csv (str): Path to people.csv.
        labels_csv (str): Path to labels.csv.
        out_dir (str): Output directory for repo, data, and logs.
        sample_frac (float): Fraction of people to sample.

    Returns:
        BenchResult: Run status and parsed metric.
    """
    import subprocess, sys as _sys
    os.makedirs(out_dir, exist_ok=True)
    data_root = os.path.join(out_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    prepare_ditto_data(people_csv, labels_csv, data_root, max_pos=1000, sample_frac=sample_frac, seed=42)
    repo = os.path.join(out_dir, "repo")
    if not os.path.exists(repo):
        try:
            subprocess.check_call([_sys.executable, "-m", "pip", "install", "gitpython"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        try:
            subprocess.check_call(["git", "clone", "https://github.com/megagonlabs/ditto", repo])
        except Exception as e:
            return BenchResult(name="Ditto", status="fail", metric=None, details={"error": f"git clone failed: {e}"})
    try:
        dp = os.path.join(repo, "ditto_light", "ditto.py")
        if os.path.exists(dp):
            s = open(dp, "r").read()
            s = s.replace("from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup", "from transformers import AutoModel\nfrom transformers.optimization import get_linear_schedule_with_warmup\nimport torch.optim as optim")
            s = s.replace("optimizer = AdamW(", "optimizer = optim.AdamW(")
            open(dp, "w").write(s)
    except Exception:
        pass
    try:
        subprocess.run([_sys.executable, "-m", "pip", "install", "spacy"], check=False)
        subprocess.run([_sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
    except Exception:
        pass
    log_fp = os.path.join(out_dir, "train.log")
    # write configs.json mapping ICEID task to our files
    try:
        cfg = [{
            "name": "ICEID",
            "trainset": os.path.join("ICEID", "train.txt"),
            "validset": os.path.join("ICEID", "valid.txt"),
            "testset": os.path.join("ICEID", "test.txt"),
        }]
        with open(os.path.join(repo, "configs.json"), "w") as f:
            json.dump(cfg, f)
    except Exception:
        pass
    cmd = [
        _sys.executable, "-u", "train_ditto.py",
        "--task", "ICEID",
        "--batch_size", "16",
        "--max_len", "64",
        "--lr", "3e-5",
        "--n_epochs", "1",
        "--lm", "distilbert",
        "--summarize",
    ]
    env = os.environ.copy()
    env["DITTO_DATA"] = data_root
    try:
        p = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
        with open(log_fp, "w") as f:
            f.write(p.stdout)
            f.write("\n\n--- STDERR ---\n\n")
            f.write(p.stderr)
        import re
        m = re.search(r"F1(?:-score)?:\s*([0-9.]+)", p.stdout, re.IGNORECASE)
        metric = f"F1={m.group(1)}" if m else None
        status = "ok" if p.returncode == 0 and metric else ("fail")
        return BenchResult(name="Ditto", status=status, metric=metric, details={"log": log_fp})
    except Exception as e:
        return BenchResult(name="Ditto", status="fail", metric=None, details={"error": str(e), "log": log_fp})


def _load_dm_splits(dataset_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[int, str]]:
    """Loads DeepMatcher-style splits and text maps.

    Args:
        dataset_dir (str): Directory with left/right and split CSVs.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[int, str]]: Train/valid/test and id→text maps.
    """
    manifest_path=Path(dataset_dir)/'split_manifest.json'
    release_path=Path(dataset_dir).resolve().parent/'manifest.json'
    if release_path.exists() and json.loads(release_path.read_text()).get('person_column_authoritative') and not manifest_path.exists():
        raise ValueError('Selected-release pair export is incomplete: missing final split manifest')
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if 'output_sha256' in manifest:
            if manifest.get('status')!='complete' or len(manifest['output_sha256'])!=5:
                raise ValueError('Pair export is incomplete')
            for name,digest in manifest['output_sha256'].items():
                if _sha256_file(str(Path(dataset_dir)/name))!=digest:raise ValueError('Pair export changed after generation: '+name)
    left = pd.read_csv(os.path.join(dataset_dir, "left.csv"))
    right = pd.read_csv(os.path.join(dataset_dir, "right.csv"))
    lmap = left.set_index("id")["text"].to_dict()
    rmap = right.set_index("id")["text"].to_dict()
    train = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    valid = pd.read_csv(os.path.join(dataset_dir, "validation.csv"))
    test = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
    return train, valid, test, lmap, rmap


def run_hf_classifier(dataset_dir: str, out_dir: str) -> BenchResult:
    """Fine-tunes a HuggingFace transformer for pair classification.

    Args:
        dataset_dir (str): Directory with DeepMatcher-style splits.
        out_dir (str): Directory for artifacts.

    Returns:
        BenchResult: Status and F1 metric if available.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        from sklearn.metrics import f1_score
    except Exception as e:
        try:
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "transformers", "torch"], check=False)
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
            from sklearn.metrics import f1_score
        except Exception as e2:
            return BenchResult(name="HF_DistilBERT", status="fail", metric=None, details={"error": str(e2)})
    try:
        os.makedirs(out_dir, exist_ok=True)
        train, valid, test, lmap, rmap = _load_dm_splits(dataset_dir)
        def to_pairs(df: pd.DataFrame):
            texts = (df["ltable_id"].map(lmap).fillna("") + " [SEP] " + df["rtable_id"].map(rmap).fillna(""))
            labels = df["label"].astype(int).tolist()
            return texts.tolist(), labels
        tr_texts, tr_labels = to_pairs(train)
        va_texts, va_labels = to_pairs(valid)
        te_texts, te_labels = to_pairs(test)
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        def tokenize(texts):
            return tok(texts, padding=True, truncation=True, max_length=128)
        tr_enc = tokenize(tr_texts)
        va_enc = tokenize(va_texts)
        te_enc = tokenize(te_texts)
        class SimpleDS(torch.utils.data.Dataset):
            def __init__(self, enc, labels):
                self.enc = enc
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            learning_rate=3e-5,
            logging_steps=50,
            lr_scheduler_type="linear",
            report_to=[],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        trainer = Trainer(model=model, args=args, train_dataset=SimpleDS(tr_enc, tr_labels))
        trainer.train()
        with torch.no_grad():
            inp = {k: torch.tensor(v).to(device) for k, v in te_enc.items()}
            logits = trainer.model(**inp).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
        f1 = f1_score(te_labels, preds)
        return BenchResult(name="HF_DistilBERT", status="ok", metric=f"F1={f1:.4f}")
    except Exception as e:
        try:
            with open(os.path.join(out_dir, "error.txt"), "w") as f:
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        return BenchResult(name="HF_DistilBERT", status="fail", metric=None, details={"error": str(e)})


def run_sbert_zeroer(dataset_dir: str, out_dir: str) -> BenchResult:
    """Uses Sentence-BERT embeddings with cosine similarity and a tuned threshold.

    Args:
        dataset_dir (str): Directory with DeepMatcher-style splits.
        out_dir (str): Directory for artifacts.

    Returns:
        BenchResult: Status and F1 metric if available.
    """
    try:
        import numpy as _np
        from sklearn.metrics import f1_score
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:
        try:
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "pip", "install", "sentence-transformers"], check=False)
            import numpy as _np
            from sklearn.metrics import f1_score
            from sentence_transformers import SentenceTransformer, util
        except Exception as e2:
            return BenchResult(name="SBERT_ZeroER", status="fail", metric=None, details={"error": str(e2)})
    try:
        os.makedirs(out_dir, exist_ok=True)
        train, valid, test, lmap, rmap = _load_dm_splits(dataset_dir)
        def enc_pairs(df):
            lt = df["ltable_id"].map(lmap).fillna("").tolist()
            rt = df["rtable_id"].map(rmap).fillna("").tolist()
            y = df["label"].astype(int).tolist()
            return lt, rt, y
        vlt, vrt, vy = enc_pairs(valid)
        tlt, trt, ty = enc_pairs(test)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vecl = model.encode(vlt, convert_to_tensor=True, show_progress_bar=False)
        vecr = model.encode(vrt, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(vecl, vecr).diagonal().cpu().numpy()
        best_f1 = 0.0
        best_t = 0.5
        for t in np.linspace(0.3, 0.9, 25):
            preds = (sims >= t).astype(int)
            f1 = f1_score(vy, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        te_l = model.encode(tlt, convert_to_tensor=True, show_progress_bar=False)
        te_r = model.encode(trt, convert_to_tensor=True, show_progress_bar=False)
        te_s = util.cos_sim(te_l, te_r).diagonal().cpu().numpy()
        te_pred = (te_s >= best_t).astype(int)
        f1 = f1_score(ty, te_pred)
        return BenchResult(name="SBERT_ZeroER", status="ok", metric=f"F1={f1:.4f}")
    except Exception as e:
        try:
            with open(os.path.join(out_dir, "error.txt"), "w") as f:
                f.write(str(e) + "\n\n")
                f.write(traceback.format_exc())
        except Exception:
            pass
        return BenchResult(name="SBERT_ZeroER", status="fail", metric=None, details={"error": str(e)})


def run_ditto_reimpl(dataset_dir: str, out_dir: str) -> BenchResult:
    """Reimplements Ditto-style PLM classification using the HF classifier pipeline above.

    Args:
        dataset_dir (str): Directory with DeepMatcher-style splits (left/right + train/valid/test).
        out_dir (str): Output directory.

    Returns:
        BenchResult: Status and F1 metric if available.
    """
    try:
        return run_hf_classifier(dataset_dir, out_dir)
    except Exception as e:
        return BenchResult(name="Ditto_Reimpl", status="fail", metric=None, details={"error": str(e)})


def run_deepmatcher_isolated(dataset_dir: str, out_dir: str) -> BenchResult:
    return BenchResult(name="DeepMatcher(iso)", status="fail", metric=None, details={"error": "removed"})


def _try_external_models() -> List[Dict[str, str]]:
    """Attempts to run Ditto and ZeroER if available; otherwise returns skip rows.

    Returns:
        List[Dict[str, str]]: Rows for the results table for external models.
    """
    return []


