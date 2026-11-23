# other_models/dashboard/ds_io.py
"""I/O utilities for the dashboard (load people, edges, clusters, versions)
plus cached relationship indexes for fast family lookups.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
import os
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from .schemas import PeopleSchema, EdgeSchema, ClusterSchema, schema_signature


# -----------------------------
# Helpers
# -----------------------------
@lru_cache(maxsize=256)
def _content_digest(path: str,size: int,mtime_ns: int,ctime_ns: int,inode: int) -> str:
    from .eval_api import _sha256_file
    return _sha256_file(path)


def _file_sig(path: str) -> str:
    """Content-bound cache key, memoized only while exact file metadata is stable."""
    try:
        path=os.path.abspath(path);before=os.stat(path)
        identity=lambda s:(s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino)
        digest=_content_digest(path,*identity(before))
        if identity(before)!=identity(os.stat(path)):raise ValueError('Input changed while reading its content identity')
        return f"{path}|sha256:{digest}"
    except FileNotFoundError:
        return f"{path}|missing"


def canonicalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures edges are undirected (id1 < id2) and removes duplicates.

    This standardizes edge representation, making lookups and joins consistent.

    Args:
        df (pd.DataFrame): A DataFrame with 'id1' and 'id2' columns.

    Returns:
        pd.DataFrame: A new DataFrame with canonicalized and unique edges.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["id1", "id2"])
    out = df[["id1", "id2"]].copy()
    a = out["id1"].astype("int64", copy=False)
    b = out["id2"].astype("int64", copy=False)
    id1 = np.minimum(a, b)
    id2 = np.maximum(a, b)
    out.loc[:, "id1"] = id1
    out.loc[:, "id2"] = id2
    out = out.drop_duplicates(ignore_index=True)
    return out


def _parse_members_column(col: pd.Series) -> pd.Series:
    """Parses a cluster members column into lists of integers.

    Handles string formats like '1;2;3' and list-like strings '[1, 2, 3]'.

    Args:
        col (pd.Series): The DataFrame column containing cluster members.

    Returns:
        pd.Series: A Series where each element is a list of integer member IDs.
    """
    def parse_one(x):
        if isinstance(x, list):
            return [int(i) for i in x]
        if isinstance(x, (np.ndarray, tuple)):
            return [int(i) for i in list(x)]
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            return [int(x)]
        if isinstance(x, (float, np.floating)) and np.isfinite(x) and float(x).is_integer():
            return [int(x)]
        if isinstance(x, str):
            s = x.strip()
            if ";" in s and "[" not in s:
                return [int(i) for i in s.split(";") if i]
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple)):
                    return [int(i) for i in v]
                if isinstance(v, int) and not isinstance(v, bool):
                    return [v]
            except Exception:
                pass
        return []
    return col.apply(parse_one)


def members_to_str(members: List[int]) -> str:
    """Converts a list of member IDs into a semicolon-separated string.

    Args:
        members (List[int]): A list of integer IDs.

    Returns:
        str: A string like "id1;id2;id3".
    """
    return ";".join(str(i) for i in members)


def _validate_cluster_members(clusters: pd.DataFrame) -> int:
    """Validate a parsed cluster table as a disjoint record partition, in place.

    Members are a set: a record repeated inside one cluster is collapsed and the
    declared size is corrected (legacy exports listed every member twice and
    counted both). A record in two different clusters is still rejected.

    Returns:
        int: Number of clusters whose repeated member entries were collapsed.
    """
    unique = clusters["members"].map(lambda group: list(dict.fromkeys(group)))
    repeated = unique.map(len).ne(clusters["members"].map(len))
    if repeated.any():
        declared = clusters.loc[repeated, "size"]
        if not (declared.eq(clusters.loc[repeated, "members"].map(len)) |
                declared.eq(unique[repeated].map(len))).all():
            raise ValueError("Cluster member lists disagree with declared sizes; refusing a lossy load")
        clusters["members"] = unique
        clusters["size"] = unique.map(len).astype(clusters["size"].dtype)
    if not clusters["size"].eq(clusters["members"].map(len)).all():
        raise ValueError("Cluster member lists disagree with declared sizes; refusing a lossy load")
    members = [member for group in clusters["members"] for member in group]
    if len(members) != len(set(members)) or clusters["cluster_id"].duplicated().any():
        raise ValueError("Cluster output is not a disjoint record partition")
    return int(repeated.sum())


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_parish_map(parish_csv: str, file_signature: str) -> Dict[int, str]:
    """Loads and caches a mapping from parish ID to parish name.

    Args:
        parish_csv (str): The path to the parish CSV file.
        file_signature (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        Dict[int, str]: A dictionary mapping integer parish IDs to their string names.
    """
    if file_signature.endswith('|missing'):
        return {}
    df = pd.read_csv(parish_csv, usecols=["id", "name"], dtype={"id": "Int64", "name": str})
    return pd.Series(df.name.values, index=df.id).to_dict()

@st.cache_data(show_spinner=False)
def load_people(people_csv: str, file_signature: str, schema: Optional[PeopleSchema] = None, schema_sig: str = "none") -> pd.DataFrame:
    """Loads and caches the main people.csv file with necessary type coercion.

    This function selects a subset of columns, converts IDs and years to nullable integers,
    and normalizes key string fields to lowercase.

    Args:
        people_csv (str): Path to the people.csv file.
        file_signature (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        pd.DataFrame: The loaded and preprocessed people data.
    """
    if file_signature.endswith('|missing'):
        raise FileNotFoundError(f"people.csv not found: {people_csv}")

    usecols = [
        "id", "heimild", "nafn_norm", "first_name", "middle_name", "patronym", "surname",
        "birthyear", "sex", "status", "marriagestatus", "person", "partner", "father", "mother",
        "farm", "county", "parish", "district",
        'source_father','source_mother','source_partner','source_farm'
    ]
    df = pd.read_csv(people_csv, dtype=str, keep_default_na=False, low_memory=False)
    if schema is not None and schema.rename_map:
        df = df.rename(columns=schema.rename_map)
    for c in usecols:
        if c not in df.columns:
            df[c] = ""
    df = df[usecols].copy()

    # numerics to Int64
    for numc in ["id", "heimild", "birthyear", "farm", "county", "parish", "district", "partner", "father", "mother", "person",
                 'source_father','source_mother','source_partner','source_farm']:
        df[numc] = pd.to_numeric(df[numc], errors="coerce").astype("Int64")

    # normalize strings
    for sc in ["nafn_norm", "first_name", "middle_name", "patronym", "surname", "sex", "status", "marriagestatus"]:
        df[sc] = df[sc].astype(str).str.lower()

    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64", copy=False)
    from .eval_api import _sha256_file
    df.attrs['source_signature']=_sha256_file(people_csv)+schema_signature(schema)
    return df


def detect_backends(run_root: str) -> list[str]:
    """Scans a directory to find valid backend subdirectories.

    A valid backend is a subdirectory that contains a 'clusters.csv' file.

    Args:
        run_root (str): The path to the parent directory (e.g., a shard directory).

    Returns:
        list[str]: A sorted list of detected backend names.
    """
    if not os.path.isdir(run_root):
        return []
    subs = []
    for name in os.listdir(run_root):
        p = os.path.join(run_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "clusters.csv")):
            subs.append(name)
    subs.sort()
    return subs


@st.cache_data(show_spinner=False)
def _read_clusters_cached(cl_path: str, file_signature: str) -> pd.DataFrame:
    """Reads and caches a clusters.csv file, parsing the 'members' column.

    Args:
        cl_path (str): The path to the clusters.csv file.
        file_signature (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        pd.DataFrame: The loaded cluster data with a 'members' column of type list[int].
    """
    clusters = pd.read_csv(cl_path, dtype={"cluster_id": "Int64", "size": "Int64", "members": str},
                           keep_default_na=False, low_memory=False)
    clusters["members"] = _parse_members_column(clusters.get("members", pd.Series([], dtype=str)))
    clusters["members"] = clusters["members"].apply(lambda x: list(map(int, x)))
    _validate_cluster_members(clusters)
    return clusters


@st.cache_data(show_spinner=False)
def _read_edges_cached(ed_path: str, file_signature: str) -> pd.DataFrame:
    """Reads and caches an edges.csv file, ensuring edges are canonicalized.

    Args:
        ed_path (str): The path to the edges.csv file.
        file_signature (str): A file signature used by Streamlit for cache invalidation.

    Returns:
        pd.DataFrame: The loaded and canonicalized edge data.
    """
    if not os.path.exists(ed_path):
        return pd.DataFrame(columns=["id1", "id2"])
    edges = pd.read_csv(ed_path, dtype={"id1": "Int64", "id2": "Int64"}, low_memory=False)
    edges = edges.dropna(subset=["id1", "id2"]).copy()
    edges["id1"] = edges["id1"].astype("int64", copy=False)
    edges["id2"] = edges["id2"].astype("int64", copy=False)
    edges = canonicalize_edges(edges)
    return edges


def load_edges_clusters(backend_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads edges.csv and clusters.csv from a specific backend directory.

    Args:
        backend_dir (str): The path to the directory containing the files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the edges DataFrame and the clusters DataFrame.
    """
    ed_path = os.path.join(backend_dir, "edges.csv")
    cl_path = os.path.join(backend_dir, "clusters.csv")
    if not os.path.exists(cl_path):
        raise FileNotFoundError(f"clusters.csv not found in {backend_dir}")
    clusters = _read_clusters_cached(cl_path, _file_sig(cl_path))
    edges = _read_edges_cached(ed_path, _file_sig(ed_path))
    return edges, clusters


@st.cache_data(show_spinner=False)
def _read_edges_cached_with_schema(ed_path: str, file_signature: str, schema: Optional[EdgeSchema] = None) -> pd.DataFrame:
    """Reads an edges file and applies an optional schema for column names.

    Args:
        ed_path (str): The path to the edges file.
        file_signature (str): A file signature used for cache invalidation.
        schema (Optional[EdgeSchema]): Optional column mapping for id1/id2.

    Returns:
        pd.DataFrame: The loaded and canonicalized edge data.
    """
    if schema is None:
        return _read_edges_cached(ed_path, file_signature)
    if not os.path.exists(ed_path):
        return pd.DataFrame(columns=["id1", "id2"])
    edges = pd.read_csv(ed_path, low_memory=False)
    c1 = schema.id1_col
    c2 = schema.id2_col
    if c1 not in edges.columns or c2 not in edges.columns:
        return pd.DataFrame(columns=["id1", "id2"])
    edges = edges.rename(columns={c1: "id1", c2: "id2"})
    edges = edges.dropna(subset=["id1", "id2"]).copy()
    edges["id1"] = pd.to_numeric(edges["id1"], errors="coerce").astype("Int64")
    edges["id2"] = pd.to_numeric(edges["id2"], errors="coerce").astype("Int64")
    edges = edges.dropna(subset=["id1", "id2"]).copy()
    edges["id1"] = edges["id1"].astype("int64", copy=False)
    edges["id2"] = edges["id2"].astype("int64", copy=False)
    return canonicalize_edges(edges)


@st.cache_data(show_spinner=False)
def _read_clusters_cached_with_schema(cl_path: str, file_signature: str, schema: Optional[ClusterSchema] = None) -> pd.DataFrame:
    """Reads a clusters file and applies an optional schema for columns and members.

    Args:
        cl_path (str): The path to the clusters file.
        file_signature (str): A file signature used for cache invalidation.
        schema (Optional[ClusterSchema]): Optional column mapping and members separator.

    Returns:
        pd.DataFrame: The loaded cluster data with canonical column names.
    """
    if schema is None:
        return _read_clusters_cached(cl_path, file_signature)
    clusters = pd.read_csv(cl_path, low_memory=False)
    needed = [schema.cluster_id_col, schema.size_col, schema.members_col]
    for c in needed:
        if c not in clusters.columns:
            clusters[c] = [] if c == schema.members_col else pd.NA
    clusters = clusters.rename(columns={
        schema.cluster_id_col: "cluster_id",
        schema.size_col: "size",
        schema.members_col: "members",
    })
    if schema.members_sep:
        clusters["members"] = clusters["members"].astype(str).apply(lambda s: [int(x) for x in s.split(schema.members_sep) if x])
    else:
        clusters["members"] = _parse_members_column(clusters.get("members", pd.Series([], dtype=str)))
    clusters["members"] = clusters["members"].apply(lambda x: list(map(int, x)))
    clusters["cluster_id"] = pd.to_numeric(clusters["cluster_id"], errors="coerce").astype("Int64")
    clusters["size"] = pd.to_numeric(clusters["size"], errors="coerce").astype("Int64")
    return clusters


def load_edges_clusters_with_schema(backend_dir: str, edge_schema: Optional[EdgeSchema] = None, cluster_schema: Optional[ClusterSchema] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads edges and clusters with optional schemas for column adaptation.

    Args:
        backend_dir (str): The directory containing edges and clusters files.
        edge_schema (Optional[EdgeSchema]): Optional schema for edges columns.
        cluster_schema (Optional[ClusterSchema]): Optional schema for cluster columns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A pair of dataframes for edges and clusters.
    """
    ed_path = os.path.join(backend_dir, "edges.csv")
    cl_path = os.path.join(backend_dir, "clusters.csv")
    if not os.path.exists(cl_path):
        raise FileNotFoundError(f"clusters.csv not found in {backend_dir}")
    edges = _read_edges_cached_with_schema(ed_path, _file_sig(ed_path), edge_schema)
    clusters = _read_clusters_cached_with_schema(cl_path, _file_sig(cl_path), cluster_schema)
    return edges, clusters


def artifact_kind(directory: str) -> str:
    """Keep reference-derived review structures distinguishable after saving."""
    kinds=set()
    for name in ['provenance.json','meta.json']:
        path=os.path.join(directory,name)
        if os.path.isfile(path):
            with open(path) as stream:kind=json.load(stream).get('artifact_kind')
            if kind and kind!='unspecified':kinds.add(kind)
    if len(kinds)>1:raise ValueError('Conflicting saved artifact roles')
    return next(iter(kinds),'unspecified')


def save_version(
    base_backend_dir: str,
    people_csv: str,
    edges: pd.DataFrame,
    clusters: pd.DataFrame,
    out_root: str,
    comment: str = "",
    version_name: str | None = None,
) -> str:
    """Saves a snapshot of edges and clusters to a new versioned directory.

    The new directory will contain edges.csv, clusters.csv, and a meta.json file
    with information about the source data and creation time.

    Args:
        base_backend_dir (str): The original backend directory being versioned.
        people_csv (str): Path to the people.csv file used.
        edges (pd.DataFrame): The DataFrame of edges to save.
        clusters (pd.DataFrame): The DataFrame of clusters to save.
        out_root (str): The parent directory where the new version folder will be created.
        comment (str, optional): A user-provided comment to store in the metadata.
        version_name (str | None, optional): An optional name for the version folder. If None, a timestamp-based name is generated.

    Returns:
        str: The path to the newly created version directory.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    clusters = clusters.copy()
    clusters["members"] = _parse_members_column(clusters["members"])
    _validate_cluster_members(clusters)
    from .eval_api import _sha256_file
    declared_ids = sorted(int(i) for members in clusters['members'] for i in members)
    binding = {'people_sha256':_sha256_file(people_csv), 'declared_nodes':len(declared_ids),
               'declared_node_ids_sha256':hashlib.sha256(json.dumps(declared_ids,separators=(',',':')).encode()).hexdigest()}
    base_name = version_name or f"version_{ts}"
    out_dir = os.path.join(out_root, base_name)
    os.makedirs(out_dir, exist_ok=False)

    meta = {
        "created_at": ts,
        "source_backend_dir": os.path.abspath(base_backend_dir),
        "people_csv": os.path.abspath(people_csv),
        "comment": comment,
        "artifact_kind": artifact_kind(base_backend_dir),
        "binding": binding,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    e = canonicalize_edges(edges)
    e.to_csv(os.path.join(out_dir, "edges.csv"), index=False)

    c = clusters.copy()
    c = c[["cluster_id", "size", "members"]].copy()
    c["members"] = c["members"].apply(lambda m: ";".join(map(str, m)) if isinstance(m, list) else str(m))
    c.to_csv(os.path.join(out_dir, "clusters.csv"), index=False)

    return out_dir


def load_saved_version(version_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads edges and clusters from a previously saved version directory.

    Args:
        version_dir (str): The path to the version directory.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the loaded edges and clusters DataFrames.
    """
    ed_path = os.path.join(version_dir, "edges.csv")
    cl_path = os.path.join(version_dir, "clusters.csv")
    if not os.path.exists(cl_path):
        raise FileNotFoundError(f"clusters.csv missing in {version_dir}")
    clusters = _read_clusters_cached(cl_path, _file_sig(cl_path))
    edges = _read_edges_cached(ed_path, _file_sig(ed_path))
    return edges, clusters


def free_memory(*objs) -> None:
    """Explicitly deletes objects and triggers Python's garbage collector.

    This is a utility to help manage memory when dealing with large DataFrames.

    Args:
        *objs: A variable number of objects to delete.
    """
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()


# -----------------------------
# Cached relationship indexes
# -----------------------------
@st.cache_resource(show_spinner=False)
def family_record_index(people_df: pd.DataFrame, source_signature: str = '') -> Dict:
    """Resolve kin claims to source records without conflating person/row IDs.

    An exact source-record pointer takes precedence. Without one, a person
    reference resolves only to a unique same-wave appearance or a unique
    appearance overall. Ambiguous/unavailable targets stay unresolved. These
    are recorded roles, which may describe foster/household relationships.
    """
    columns=['id','person','heimild','father','mother','partner',
             'source_father','source_mother','source_partner']
    base=people_df.reindex(columns=columns).copy()
    for c in columns:base[c]=pd.to_numeric(base[c],errors='coerce').astype('Int64')
    if base.id.isna().any() or base.id.duplicated().any():raise ValueError('Family graph requires unique source-record IDs')
    base=base.set_index('id',drop=False);known=set(map(int,base.index))
    labelled=base[base.person.notna()][['id','person','heimild']]
    same=labelled.dropna(subset=['heimild']).drop_duplicates(['person','heimild'],keep=False).set_index(['person','heimild']).id
    single=labelled.drop_duplicates('person',keep=False).set_index('person').id
    resolved=pd.DataFrame(index=base.index);parents={};children={};partner_in={}
    for relation in ['father','mother','partner']:
        source=base['source_'+relation];reference=base[relation]
        target=source.where(source.isin(known));basis=pd.Series('',index=base.index,dtype=object)
        basis.loc[target.notna()]='source_record'
        basis.loc[source.notna()&target.isna()]='source_record_unavailable'
        target_reference=target.map(base.person)
        mismatch=target.notna()&reference.notna()&target_reference.notna()&reference.ne(target_reference)
        basis.loc[mismatch]='source_record_reference_disagrees'
        fallback=source.isna()&reference.notna()
        lookup=pd.MultiIndex.from_arrays([reference,base.heimild],names=['person','heimild'])
        same_target=pd.Series(same.reindex(lookup).to_numpy(),index=base.index,dtype='Int64')
        use=fallback&same_target.notna();target.loc[use]=same_target[use];basis.loc[use]='unique_reference_wave'
        single_target=reference.map(single)
        use=fallback&target.isna()&single_target.notna();target.loc[use]=single_target[use];basis.loc[use]='unique_reference_observation'
        basis.loc[fallback&target.isna()]='person_reference_unresolved'
        resolved[relation+'_record']=target;resolved[relation+'_basis']=basis
        out={int(a):int(b) for a,b in target.dropna().items()}
        incoming={}
        for child,parent in out.items():incoming.setdefault(parent,[]).append(child)
        if relation=='partner':partner_out=out;partner_in=incoming
        else:parents[relation]=out;children[relation]=incoming
    return {'known':known,'records':resolved,'parents':parents,'children':children,
            'partner_out':partner_out,'partner_in':partner_in}


def build_children_index(people_df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]:
    """Return existing lookup interface with IDs consistently in source namespace."""
    index=family_record_index(people_df,people_df.attrs.get('source_signature',''))
    return index['children']['father'],index['children']['mother'],index['partner_out']
