"""
Thin, importable wrapper around the standalone 'analysis' script so Streamlit
can call evaluation programmatically and display results inline.

Usage:
    from dashboard.eval_api import analyze_backends
    results = analyze_backends(run_dir, people_csv, labels_csv, backends=None)
"""

from __future__ import annotations

import json
import math
import os
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from .schemas import LabelsSchema


def _sha256_file(path: str) -> str:
    digest=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _load_gt(people_csv: str, labels_csv: str, schema: LabelsSchema | None = None) -> pd.DataFrame:
    """Load the authoritative reference mapping, preserving unknown identities.

    It creates a DataFrame mapping each record 'id' to a 'gt_cluster' ID,
    prioritizing the 'person' column from people_csv if available.

    Args:
        people_csv (str): Path to the people.csv file.
        labels_csv (str): Path to the labels CSV file (e.g., ground truth).

    Returns:
        pd.DataFrame: A DataFrame with 'id' and 'gt_cluster' columns.
    """
    people_sha256 = _sha256_file(people_csv)
    release_manifest = Path(people_csv).resolve().parent.parent / "manifest.json"
    if release_manifest.exists():
        release = json.loads(release_manifest.read_text())
        if release.get("status") == "building":
            raise ValueError("Reference release generation is incomplete")
        if release.get("person_column_authoritative"):
            if release.get("status") != "complete":
                raise ValueError("Reference release generation is incomplete")
            if people_sha256 != release["output_people_sha256"]:
                raise ValueError("Reference table does not match its release manifest")
    people = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    # An explicitly supplied person column is authoritative even for an entirely
    # unlabelled subset. Otherwise fallback can resurrect withdrawn assignments.
    person_is_authoritative = "person" in people.columns
    labels = (pd.read_csv(labels_csv, dtype=str, keep_default_na=False, low_memory=False)
              if not person_is_authoritative else pd.DataFrame())

    if schema is None:
        for c in ["id", "person"]:
            if c not in people.columns:
                people[c] = ""
        for c in ["id", "bi_einstaklingur"]:
            if c not in labels.columns:
                labels[c] = ""
    else:
        if "id" not in people.columns:
            people["id"] = ""
        if "person" not in people.columns:
            people["person"] = ""
        if schema.id_col not in labels.columns:
            labels[schema.id_col] = ""
        if schema.cluster_col not in labels.columns:
            labels[schema.cluster_col] = ""

    people["id"] = pd.to_numeric(people["id"], errors="coerce")
    people["person"] = pd.to_numeric(people["person"], errors="coerce")
    if schema is None:
        labels_id_col = "id"
        labels_cluster_col = "bi_einstaklingur"
    else:
        labels_id_col = schema.id_col
        labels_cluster_col = schema.cluster_col
    labels[labels_id_col] = pd.to_numeric(labels[labels_id_col], errors="coerce")
    labels[labels_cluster_col] = pd.to_numeric(labels[labels_cluster_col], errors="coerce")

    people = people.dropna(subset=["id"]).copy()
    people["id"] = people["id"].astype(int)
    if people["id"].duplicated().any():
        raise ValueError("Duplicate record IDs in people reference mapping")

    # Prefer 'person' if present, else labels
    gt = people[["id", "person"]].copy()
    if person_is_authoritative:
        gt["gt_cluster"] = gt["person"]
    else:
        if labels[labels_id_col].dropna().duplicated().any():
            raise ValueError("Duplicate record IDs in fallback reference mapping")
        if schema is None:
            gt = gt.merge(labels[["id", "bi_einstaklingur"]], on="id", how="left")
            gt["gt_cluster"] = gt["bi_einstaklingur"]
        else:
            tmp = labels[[labels_id_col, labels_cluster_col]].copy()
            tmp = tmp.rename(columns={labels_id_col: "id", labels_cluster_col: "gt_cluster"})
            gt = gt.merge(tmp, on="id", how="left")
    gt = gt[["id", "gt_cluster"]].copy()
    gt.attrs['reference_provenance'] = {
        'people_csv':str(Path(people_csv).resolve()), 'people_sha256':people_sha256,
        'label_source':'people.person' if person_is_authoritative else str(Path(labels_csv).resolve()),
        'reference_records':len(gt), 'known_reference_records':int(gt.gt_cluster.notna().sum()),
        'unknown_reference_records':int(gt.gt_cluster.isna().sum())}
    return gt


def _wilson_interval(p_hat: float, n: int, z: float = 1.96):
    """Calculates the Wilson score confidence interval for a binomial proportion.

    Args:
        p_hat (float): The observed proportion of successes.
        n (int): The total number of trials.
        z (float, optional): The z-score for the desired confidence level. Defaults to 1.96 (95%).

    Returns:
        tuple[Optional[float], Optional[float]]: A tuple containing the lower and upper bounds of the confidence interval.
    """
    if n == 0:
        return (None, None)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _analyze_backend(bname: str, run_dir: str, gt: pd.DataFrame) -> Dict:
    """Performs a detailed analysis for a single backend's output.

    It calculates precision on labeled data, identifies novel edges, and computes
    cluster purity and size statistics if cluster data is available.

    Args:
        bname (str): The name of the backend (subfolder name).
        run_dir (str): The parent directory containing the backend folder.
        gt (pd.DataFrame): The ground truth DataFrame with 'id' and 'gt_cluster' columns.

    Returns:
        Dict: A dictionary containing a comprehensive set of evaluation metrics.
    """
    method_dir = run_dir if bname in (".", "", None) else os.path.join(run_dir, bname)
    edges_path = os.path.join(method_dir, "edges.csv")
    clusters_path = os.path.join(method_dir, "clusters.csv")
    out = {"backend": bname if bname not in (".", "", None) else os.path.basename(method_dir) or "auto"}
    from .ds_io import artifact_kind, _parse_members_column, _validate_cluster_members
    from .edits import edges_to_clusters
    provenance_path = Path(method_dir)/'provenance.json'
    binding_path = provenance_path if provenance_path.is_file() else None
    native_provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    bound_people_sha = native_provenance.get('binding',{}).get('people_sha256')
    meta_path = Path(method_dir)/'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    saved_binding = meta.get('binding',{})
    kind = artifact_kind(method_dir)
    if kind=='model_predictions' and not native_provenance and not all(
            saved_binding.get(k) is not None for k in ['people_sha256','declared_nodes','declared_node_ids_sha256']):
        raise ValueError('Saved model version lacks its reference/population binding; save a new snapshot from the original backend')
    if bound_people_sha is None and saved_binding.get('people_sha256'):
        binding_path = meta_path
        bound_people_sha = saved_binding['people_sha256']
    if kind=='model_predictions' and bound_people_sha is None:
        raise ValueError('Model predictions lack an effective reference-table binding')
    if saved_binding.get('people_sha256') is not None and saved_binding['people_sha256']!=gt.attrs['reference_provenance']['people_sha256']:
        raise ValueError('Saved version and selected reference table have different source hashes')
    if bound_people_sha is not None and bound_people_sha!=gt.attrs['reference_provenance']['people_sha256']:
        raise ValueError('Model output and selected reference table have different source hashes')
    for name, expected in native_provenance.get('output_sha256',{}).items():
        if _sha256_file(str(Path(method_dir)/name))!=expected:
            raise ValueError('Native model export changed after generation: '+name)
    out['evaluation_provenance'] = {
        'artifact_kind':kind, 'reference':gt.attrs['reference_provenance'],
        'backend_directory':str(Path(method_dir).resolve()),
        'binding_provenance_path':str(binding_path.resolve()) if binding_path else None,
        'binding_provenance_sha256':_sha256_file(str(binding_path)) if binding_path else None,
        'bound_people_sha256':bound_people_sha,
        'saved_population_binding':saved_binding or None,
        'pair_scope':'Accepted exported edges only; the unaccepted candidate universe is unavailable here.',
        'partition_scope':'Declared partition members, conditional on known authoritative reference IDs; unknowns excluded.',
        'input_sha256':{}}

    if not os.path.exists(edges_path):
        out["error"] = f"missing {edges_path}"
        return out

    edges = pd.read_csv(edges_path)
    edges["id1"] = edges["id1"].astype(int)
    edges["id2"] = edges["id2"].astype(int)
    if edges.id1.eq(edges.id2).any():
        raise ValueError('Self-links are not valid entity-resolution edges')

    gt_map = dict(zip(gt["id"].astype(int), gt["gt_cluster"]))

    def is_lab(i):
        v = gt_map.get(int(i))
        return (v is not None) and (not pd.isna(v))

    def same_ent(a, b):
        va, vb = gt_map.get(int(a)), gt_map.get(int(b))
        return (not pd.isna(va)) and (not pd.isna(vb)) and (va == vb)

    n_total = len(edges)
    
    id1_lab = edges["id1"].apply(is_lab)
    id2_lab = edges["id2"].apply(is_lab)
    
    both_lab_mask = id1_lab & id2_lab
    exactly_one_lab_mask = id1_lab ^ id2_lab
    none_lab_mask = ~(both_lab_mask | exactly_one_lab_mask)

    pred_pairs_labeled = int(both_lab_mask.sum())
    if pred_pairs_labeled > 0:
        labeled_edges = edges[both_lab_mask]
        id1_gt = labeled_edges["id1"].astype(int).map(gt_map)
        id2_gt = labeled_edges["id2"].astype(int).map(gt_map)
        tp_labeled = int(((id1_gt == id2_gt) & id1_gt.notna() & id2_gt.notna()).sum())
    else:
        tp_labeled = 0
    fp_labeled = int(pred_pairs_labeled - tp_labeled)
    prec_labeled = tp_labeled / pred_pairs_labeled if pred_pairs_labeled else None
    # Edges share records and the reference itself is uncertain. An independent-
    # Bernoulli Wilson interval would not establish identity-accuracy coverage.
    llo, lhi = None, None

    semi_novel = int(exactly_one_lab_mask.sum())
    novel = int(none_lab_mask.sum())
    total_unlabeled_side = semi_novel + novel

    est_tp_unlabeled = None
    est_tp_unlabeled_lo = None
    est_tp_unlabeled_hi = None

    touched_nodes = pd.concat([edges["id1"], edges["id2"]]).unique()
    n_nodes = len(touched_nodes)
    touched_nodes_lab = pd.Series(touched_nodes).apply(is_lab)
    n_nodes_labeled = int(touched_nodes_lab.sum())
    n_nodes_unlabeled = n_nodes - n_nodes_labeled

    purity_weighted = None
    pct_clusters_pure90 = None
    clusters_all_unlabeled = None
    size_stats = {}
    if os.path.exists(clusters_path):
        cl = pd.read_csv(clusters_path, dtype={'members':str}, keep_default_na=False)
        cl["members"] = _parse_members_column(cl["members"])
        collapsed = _validate_cluster_members(cl)
        if collapsed:
            out['evaluation_provenance']['collapsed_repeated_member_clusters'] = collapsed
        predicted = {int(member):int(cid) for cid,members in
                     cl[['cluster_id','members']].itertuples(index=False,name=None) for member in members}
        population = sorted(predicted)
        roster_sha = hashlib.sha256(json.dumps(population,separators=(',',':')).encode()).hexdigest()
        if saved_binding and (saved_binding.get('declared_nodes')!=len(population) or
                              saved_binding.get('declared_node_ids_sha256')!=roster_sha):
            raise ValueError('Saved version lost or changed declared nodes, including isolates')
        out['evaluation_provenance']['declared_node_ids_sha256'] = roster_sha
        nodes_path = Path(method_dir)/'nodes.csv'
        if nodes_path.is_file():
            declared = pd.read_csv(nodes_path,usecols=['id']).id
            if declared.isna().any() or declared.duplicated().any() or set(declared.astype(int))!=set(population):
                raise ValueError('Declared nodes and full partition membership disagree')
        if not set(population)<=set(gt_map) or not set(touched_nodes)<=set(population):
            raise ValueError('Partition or accepted edge contains an undeclared source record')
        connected,_ = edges_to_clusters(population,list(edges[['id1','id2']].itertuples(index=False,name=None)))
        if {frozenset(x) for x in connected.values()}!={frozenset(x) for x in cl.members}:
            raise ValueError('Saved partition does not match accepted edges and declared isolates')
        known = [i for i in population if is_lab(i)]
        out.update({'partition_nodes':len(population),'partition_reference_nodes':len(known),
                    'partition_unknown_nodes':len(population)-len(known),
                    'n_isolates':int(cl['size'].eq(1).sum())})
        out['evaluation_provenance']['population_source'] = 'nodes.csv + clusters.csv' if nodes_path.is_file() else 'clusters.csv including singleton members'
        if known:
            aligned = pd.DataFrame({'truth':[int(gt_map[i]) for i in known],
                                    'pred':[predicted[i] for i in known]})
            counts = aligned.groupby(['truth','pred']).size()
            true_sizes, pred_sizes = aligned.truth.value_counts(), aligned.pred.value_counts()
            b3p = float(sum(n*n/pred_sizes[p] for (t,p),n in counts.items())/len(known))
            b3r = float(sum(n*n/true_sizes[t] for (t,p),n in counts.items())/len(known))
            out.update({'ari':float(adjusted_rand_score(aligned.truth,aligned.pred)),
                        'b3_precision':b3p,'b3_recall':b3r,
                        'b3_f1':float(2*b3p*b3r/(b3p+b3r)) if b3p+b3r else 0.0})
        sizes = cl["size"].astype(int).tolist()
        if sizes:
            size_stats = {
                "n_clusters": int(len(cl)),
                "singleton_rate": float(sum(1 for s in sizes if s == 1) / len(sizes)),
                "mean_size": float(sum(sizes) / len(sizes)),
                "p90_size": int(pd.Series(sizes).quantile(0.9)),
                "p99_size": int(pd.Series(sizes).quantile(0.99)),
                "max_size": int(max(sizes)),
            }
        purities = []
        all_unlab = 0
        for members in cl["members"]:
            labs = [gt_map.get(int(i)) for i in members if is_lab(i)]
            if len(labs) == 0:
                all_unlab += 1
                continue
            try:
                counts = Counter([int(float(x)) for x in labs if pd.notna(x)])
                if not counts:
                    continue
                maj = counts.most_common(1)[0][1]
                purities.append(maj / len(labs))
            except (ValueError, TypeError):
                continue
        if purities:
            weights = []
            vals = []
            for members in cl["members"]:
                labs = [gt_map.get(int(i)) for i in members if is_lab(i)]
                if len(labs) == 0:
                    continue
                try:
                    counts = Counter([int(float(x)) for x in labs if pd.notna(x)])
                    if not counts:
                        continue
                    maj = counts.most_common(1)[0][1]
                    vals.append(maj / len(labs))
                    weights.append(len(labs))
                except (ValueError, TypeError):
                    continue
            if weights:
                purity_weighted = float((sum(v * w for v, w in zip(vals, weights))) / sum(weights))
            pct_clusters_pure90 = float(sum(1 for v in purities if v >= 0.9) / len(purities))
        clusters_all_unlabeled = int(all_unlab)

    out.update({
        "edges_total": n_total,
        "pred_pairs_labeled": pred_pairs_labeled,
        "tp_labeled": tp_labeled,
        "fp_labeled": fp_labeled,
        "prec_labeled": float(prec_labeled) if prec_labeled is not None else None,
        "prec_labeled_wilson95": (round(llo, 6), round(lhi, 6)) if llo is not None else None,
        "semi_novel_edges": semi_novel,
        "novel_edges": novel,
        "est_true_new_edges": None if est_tp_unlabeled is None else int(round(est_tp_unlabeled)),
        "est_true_new_edges_95ci": None if est_tp_unlabeled_lo is None else (int(round(est_tp_unlabeled_lo)), int(round(est_tp_unlabeled_hi))),
        "nodes_with_any_match": n_nodes,
        "nodes_with_any_match_labeled": n_nodes_labeled,
        "nodes_with_any_match_unlabeled": n_nodes_unlabeled,
        "cluster_stats": {
            **size_stats,
            "weighted_purity_on_labeled": purity_weighted,
            "pct_clusters_pure90_on_labeled": pct_clusters_pure90,
            "clusters_all_unlabeled": clusters_all_unlabeled
        }
    })
    for name in ['edges.csv','clusters.csv','nodes.csv','meta.json','provenance.json']:
        path = Path(method_dir)/name
        if path.is_file():out['evaluation_provenance']['input_sha256'][name] = _sha256_file(str(path))
    return out


def _detect_backends(run_dir: str) -> List[str]:
    """Automatically detects backend subfolders within a run directory.

    A folder is considered a backend if it contains 'edges.csv' or 'clusters.csv'.
    If no subfolders are found, it checks if the run_dir itself is a backend.

    Args:
        run_dir (str): The path to the run directory to scan.

    Returns:
        List[str]: A sorted list of detected backend names.
    """
    cand: List[str] = []
    if not os.path.isdir(run_dir):
        return cand
    for name in os.listdir(run_dir):
        p = os.path.join(run_dir, name)
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "clusters.csv")) or os.path.exists(os.path.join(p, "edges.csv")):
                cand.append(name)
    if not cand:
        if os.path.exists(os.path.join(run_dir, "clusters.csv")) or os.path.exists(os.path.join(run_dir, "edges.csv")):
            cand = ["."]
    return sorted(cand)


def analyze_backends(run_dir: str, people_csv: str, labels_csv: str, backends: Optional[Iterable[str]] = None, labels_schema: LabelsSchema | None = None) -> Dict:
    """Runs evaluation for multiple backends and returns results as a dictionary.

    This is the main entry point for programmatically evaluating backend outputs.
    It orchestrates loading ground truth, finding backends, and analyzing each one.

    Args:
        run_dir (str): The directory containing backend subfolders.
        people_csv (str): Path to the main people.csv file.
        labels_csv (str): Path to the ground truth labels file.
        backends (Optional[Iterable[str]], optional): A specific list of backend names to
            analyze. If None, backends are auto-detected. Defaults to None.

    Returns:
        Dict: A dictionary where keys are backend names and values are their analysis results.
    """
    if backends is None:
        backends = _detect_backends(run_dir)
    backends=list(backends)
    from .ds_io import artifact_kind
    for backend in backends:
        if artifact_kind(str(Path(run_dir)/backend))=='reference_structure':
            raise ValueError('Reference-derived source-review structures cannot be evaluated as model predictions. Choose a predicted backend.')

    gt = _load_gt(people_csv, labels_csv, schema=labels_schema)
    results: Dict[str, Dict] = {}
    for b in backends:
        results[b] = _analyze_backend(b, run_dir, gt)
    # If nothing found, include a sentinel to help the UI show a message
    if not results:
        results["_debug"] = {"error": f"No backends found in {run_dir}. Expect subdirs like gbdt/, logreg/ with edges.csv (and optionally clusters.csv)."}
    return results
