#!/usr/bin/env python3
"""
Download external datasets for comparison in the ICE-ID data paper.

Usage:
    python scripts/fetch_external_datasets.py all       # Download all datasets
    python scripts/fetch_external_datasets.py febrl     # Download FEBRL only
    python scripts/fetch_external_datasets.py synthea   # Download Synthea sample
    python scripts/fetch_external_datasets.py geco3     # Download GeCo3
    python scripts/fetch_external_datasets.py orcid     # Download ORCID sample
    python scripts/fetch_external_datasets.py semparl   # Download SemParl from Zenodo
    python scripts/fetch_external_datasets.py ckcc      # Download CKCC from Zenodo
    python scripts/fetch_external_datasets.py corresp   # Download correspSearch from Zenodo
    python scripts/fetch_external_datasets.py list      # List available datasets
"""
import os
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
import tempfile
import tarfile
import zipfile
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data" / "external_datasets"
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts" / "table_data"

DATASETS = {
    "febrl": {
        "name": "FEBRL",
        "description": "Freely Extensible Biomedical Record Linkage benchmark datasets",
        "url": None,
        "type": "recordlinkage",
        "license": "BSD-like",
        "access": "Open",
        "doc_url": "https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html",
        "note": "Use recordlinkage Python package to load FEBRL datasets.",
    },
    "synthea": {
        "name": "Synthea",
        "description": "Synthetic longitudinal patient records",
        "url": "https://synthetichealth.github.io/synthea-sample-data/downloads/synthea_sample_data_csv_apr2020.zip",
        "type": "zip",
        "license": "Apache 2.0",
        "access": "Open",
        "doc_url": "https://github.com/synthetichealth/synthea",
    },
    "geco3": {
        "name": "GeCo3",
        "description": "Open synthetic data generator for record linkage",
        "url": None,
        "type": "doc_sourced",
        "license": "MIT",
        "access": "Open",
        "doc_url": "https://github.com/T-Stam/GeCo",
        "note": "Doc-sourced only; generator code must be cloned manually.",
    },
    "orcid": {
        "name": "ORCID Public Data File",
        "description": "Researcher identities over time (sample only due to size)",
        "url": None,
        "type": "api_sample",
        "license": "CC0",
        "access": "Open (full file is large)",
        "doc_url": "https://info.orcid.org/documentation/integration-guide/orcid-public-data-file/",
        "note": "Full file is ~30GB; we download a sample via API for profiling.",
    },
    "semparl": {
        "name": "SemParl / ParliamentSampo",
        "description": "Parliamentary debates KG (people, time, roles)",
        "zenodo_id": "7636420",
        "type": "zenodo",
        "license": "CC-BY-4.0",
        "access": "Open",
        "doc_url": "https://www.ldf.fi/dataset/semparl",
    },
    "ckcc": {
        "name": "CKCC Correspondence KG",
        "description": "Historical correspondences (people exchanging letters over time)",
        "zenodo_id": "6631385",
        "type": "zenodo",
        "license": "CC-BY-4.0",
        "access": "Open",
        "doc_url": "https://www.ldf.fi/dataset/ckcc",
    },
    "corresp": {
        "name": "correspSearch",
        "description": "Aggregated epistolary metadata across institutions",
        "zenodo_id": "5972316",
        "type": "zenodo",
        "license": "CC-BY-4.0",
        "access": "Open",
        "doc_url": "https://www.ldf.fi/dataset/corresp",
    },
}

IPUMS_DOC_ONLY = {
    "ipums_lrs": {
        "name": "IPUMS USA Linked Representative Samples (LRS)",
        "description": "Longitudinally linked U.S. census microdata",
        "doc_url": "https://usa.ipums.org/usa/lrs.shtml",
        "license": "Restricted (account required)",
        "access": "Account required",
        "time_span": "1850-1940",
        "n_records": "~50M linked records",
        "note": "Doc-sourced only; requires IPUMS account.",
    },
    "ipums_mlp": {
        "name": "IPUMS USA Multigenerational Longitudinal Panel (MLP)",
        "description": "Multigenerational panel built from linked census and other sources",
        "doc_url": "https://usa.ipums.org/usa/mlp.shtml",
        "license": "Restricted (account required)",
        "access": "Account required",
        "time_span": "1870-2020",
        "n_records": "~100M records",
        "note": "Doc-sourced only; requires IPUMS account.",
    },
    "ipums_napp": {
        "name": "IPUMS International NAPP",
        "description": "North Atlantic Population Project historical census microdata",
        "doc_url": "https://international.ipums.org/international/napp.shtml",
        "license": "Restricted (account required)",
        "access": "Account required",
        "time_span": "1801-1910",
        "n_records": "~100M records across countries",
        "note": "Doc-sourced only; requires IPUMS account.",
    },
}


def download_file(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=300, allow_redirects=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


def download_zenodo_record(record_id: str, dest_dir: Path) -> bool:
    """Download all files from a Zenodo record."""
    api_url = f"https://zenodo.org/api/records/{record_id}"
    
    try:
        print(f"  Fetching Zenodo record {record_id}...")
        response = requests.get(api_url, timeout=60)
        response.raise_for_status()
        record = response.json()
        
        files = record.get("files", [])
        if not files:
            print(f"  No files found in Zenodo record {record_id}")
            return False
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        for file_info in files:
            file_url = file_info.get("links", {}).get("self")
            filename = file_info.get("key", "unknown")
            file_size = file_info.get("size", 0)
            
            if not file_url:
                continue
            
            dest_path = dest_dir / filename
            if dest_path.exists() and dest_path.stat().st_size == file_size:
                print(f"  {filename} already exists, skipping")
                continue
            
            print(f"  Downloading {filename} ({file_size / 1e6:.1f} MB)")
            if not download_file(file_url, dest_path, desc=filename):
                return False
        
        # Save record metadata
        with open(dest_dir / "zenodo_record.json", "w") as f:
            json.dump(record, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"  Error fetching Zenodo record: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> bool:
    """Extract tar.gz or zip archive."""
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        if str(archive_path).endswith('.tar.gz') or str(archive_path).endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(dest_dir)
        elif str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
        else:
            print(f"  Unknown archive format: {archive_path}")
            return False
        
        return True
    except Exception as e:
        print(f"  Error extracting: {e}")
        return False


def fetch_febrl() -> Dict[str, Any]:
    """Load FEBRL datasets via recordlinkage package."""
    ds_dir = DATA_DIR / "febrl"
    raw_dir = ds_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    info = DATASETS["febrl"].copy()
    info["fetched_at"] = datetime.now().isoformat()
    info["status"] = "pending"
    
    try:
        import recordlinkage
        from recordlinkage.datasets import load_febrl1, load_febrl2, load_febrl3, load_febrl4
        
        print("  Loading FEBRL datasets via recordlinkage...")
        
        datasets_loaded = []
        
        # Load FEBRL1
        try:
            df1 = load_febrl1()
            df1.to_csv(raw_dir / "febrl1.csv")
            datasets_loaded.append("febrl1")
        except Exception as e:
            print(f"    FEBRL1 failed: {e}")
        
        # Load FEBRL2
        try:
            df2 = load_febrl2()
            df2.to_csv(raw_dir / "febrl2.csv")
            datasets_loaded.append("febrl2")
        except Exception as e:
            print(f"    FEBRL2 failed: {e}")
        
        # Load FEBRL3
        try:
            df3 = load_febrl3()
            df3.to_csv(raw_dir / "febrl3.csv")
            datasets_loaded.append("febrl3")
        except Exception as e:
            print(f"    FEBRL3 failed: {e}")
        
        # Load FEBRL4
        try:
            df4_a, df4_b = load_febrl4()
            df4_a.to_csv(raw_dir / "febrl4_a.csv")
            df4_b.to_csv(raw_dir / "febrl4_b.csv")
            datasets_loaded.append("febrl4")
        except Exception as e:
            print(f"    FEBRL4 failed: {e}")
        
        info["datasets_loaded"] = datasets_loaded
        info["n_datasets"] = len(datasets_loaded)
        info["status"] = "success" if datasets_loaded else "failed"
        
    except ImportError:
        info["status"] = "failed"
        info["error"] = "recordlinkage package not installed"
    except Exception as e:
        info["status"] = "failed"
        info["error"] = str(e)
    
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return info


def fetch_synthea() -> Dict[str, Any]:
    """Download Synthea sample data."""
    ds_dir = DATA_DIR / "synthea"
    raw_dir = ds_dir / "raw"
    
    info = DATASETS["synthea"].copy()
    info["fetched_at"] = datetime.now().isoformat()
    info["status"] = "pending"
    
    archive_path = raw_dir / "synthea_sample.zip"
    
    if not archive_path.exists():
        print("  Downloading Synthea sample...")
        if not download_file(info["url"], archive_path, "Synthea"):
            info["status"] = "failed"
            return info
    
    print("  Extracting Synthea...")
    if not extract_archive(archive_path, raw_dir):
        info["status"] = "failed"
        return info
    
    csv_files = list(raw_dir.rglob("*.csv"))
    info["n_csv_files"] = len(csv_files)
    info["csv_files"] = [str(f.relative_to(raw_dir)) for f in csv_files[:20]]
    info["status"] = "success"
    
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return info


def fetch_geco3() -> Dict[str, Any]:
    """Download GeCo3 repository."""
    ds_dir = DATA_DIR / "geco3"
    raw_dir = ds_dir / "raw"
    
    info = DATASETS["geco3"].copy()
    info["fetched_at"] = datetime.now().isoformat()
    info["status"] = "pending"
    
    archive_path = raw_dir / "geco3.zip"
    
    if not archive_path.exists():
        print("  Downloading GeCo3...")
        if not download_file(info["url"], archive_path, "GeCo3"):
            info["status"] = "failed"
            return info
    
    print("  Extracting GeCo3...")
    if not extract_archive(archive_path, raw_dir):
        info["status"] = "failed"
        return info
    
    py_files = list(raw_dir.rglob("*.py"))
    info["n_py_files"] = len(py_files)
    info["status"] = "success"
    info["note"] = "Generator code downloaded; run separately to generate synthetic data."
    
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return info


def fetch_orcid() -> Dict[str, Any]:
    """Fetch ORCID sample via public API."""
    ds_dir = DATA_DIR / "orcid"
    raw_dir = ds_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    info = DATASETS["orcid"].copy()
    info["fetched_at"] = datetime.now().isoformat()
    info["status"] = "pending"
    
    # ORCID full file is ~30GB; we'll create a sample via the public API
    # Fetch a small set of public records for profiling
    sample_file = raw_dir / "orcid_sample.json"
    
    if sample_file.exists():
        print("  ORCID sample already exists")
        info["status"] = "success"
        with open(ds_dir / "meta.json", "w") as f:
            json.dump(info, f, indent=2)
        return info
    
    print("  Fetching ORCID sample via public API...")
    
    # Use ORCID public API to get some sample records
    # Note: This is a simplified sample; the full public data file is much larger
    sample_records = []
    
    # Search for some public records
    search_url = "https://pub.orcid.org/v3.0/search/"
    headers = {"Accept": "application/json"}
    
    try:
        # Get a sample of records
        params = {"q": "affiliation-org-name:university", "rows": 100}
        response = requests.get(search_url, params=params, headers=headers, timeout=60)
        
        if response.status_code == 200:
            results = response.json()
            orcid_ids = [r.get("orcid-identifier", {}).get("path") for r in results.get("result", [])]
            
            # Fetch details for first 20
            for orcid_id in orcid_ids[:20]:
                if orcid_id:
                    record_url = f"https://pub.orcid.org/v3.0/{orcid_id}/record"
                    rec_response = requests.get(record_url, headers=headers, timeout=30)
                    if rec_response.status_code == 200:
                        sample_records.append(rec_response.json())
            
            with open(sample_file, "w") as f:
                json.dump(sample_records, f, indent=2)
            
            info["n_sample_records"] = len(sample_records)
            info["status"] = "success"
        else:
            info["status"] = "partial"
            info["error"] = f"API returned {response.status_code}"
    
    except Exception as e:
        info["status"] = "failed"
        info["error"] = str(e)
    
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return info


def fetch_zenodo_dataset(key: str) -> Dict[str, Any]:
    """Fetch a dataset from Zenodo."""
    ds_info = DATASETS[key]
    ds_dir = DATA_DIR / key
    raw_dir = ds_dir / "raw"
    
    info = ds_info.copy()
    info["fetched_at"] = datetime.now().isoformat()
    info["status"] = "pending"
    
    zenodo_id = ds_info["zenodo_id"]
    
    print(f"  Fetching {ds_info['name']} from Zenodo record {zenodo_id}...")
    
    if download_zenodo_record(zenodo_id, raw_dir):
        info["status"] = "success"
        
        # List downloaded files
        files = list(raw_dir.glob("*"))
        info["downloaded_files"] = [f.name for f in files if f.is_file()]
    else:
        info["status"] = "failed"
    
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2)
    
    return info


FETCHERS = {
    "febrl": fetch_febrl,
    "synthea": fetch_synthea,
    "geco3": fetch_geco3,
    "orcid": fetch_orcid,
    "semparl": lambda: fetch_zenodo_dataset("semparl"),
    "ckcc": lambda: fetch_zenodo_dataset("ckcc"),
    "corresp": lambda: fetch_zenodo_dataset("corresp"),
}


def write_manifest(results: Dict[str, Any]):
    """Write manifest of all datasets."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "datasets": results,
        "doc_sourced_only": IPUMS_DOC_ONLY,
    }
    
    with open(ARTIFACTS_DIR / "external_datasets_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest written to {ARTIFACTS_DIR / 'external_datasets_manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description="Fetch external datasets")
    parser.add_argument("target", nargs="?", default="list",
                        help="Dataset to fetch: all, list, or specific dataset key")
    args = parser.parse_args()
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.target == "list":
        print("\nAvailable datasets:")
        print("-" * 60)
        for key, info in DATASETS.items():
            print(f"  {key:12} - {info['name']}")
        print("\nDoc-sourced only (require account):")
        for key, info in IPUMS_DOC_ONLY.items():
            print(f"  {key:12} - {info['name']}")
        print("\nUsage: python fetch_external_datasets.py <dataset_key|all>")
        return
    
    results = {}
    
    if args.target == "all":
        targets = list(FETCHERS.keys())
    else:
        if args.target not in FETCHERS:
            print(f"Unknown dataset: {args.target}")
            print(f"Available: {list(FETCHERS.keys())}")
            return
        targets = [args.target]
    
    print("=" * 60)
    print("FETCHING EXTERNAL DATASETS")
    print("=" * 60)
    
    for key in targets:
        print(f"\n>>> {DATASETS[key]['name']}")
        results[key] = FETCHERS[key]()
        print(f"    Status: {results[key]['status']}")
    
    write_manifest(results)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    
    # Summary
    success = sum(1 for r in results.values() if r.get("status") == "success")
    print(f"\nSuccessfully fetched: {success}/{len(results)}")


if __name__ == "__main__":
    main()

