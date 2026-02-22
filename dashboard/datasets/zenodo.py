"""
Zenodo 13 dataset provider.

Downloads and provides access to the 13 established ER benchmark datasets
from Zenodo: https://zenodo.org/records/8224111
"""

import os
import json
import zipfile
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

from .base import DatasetProvider, DatasetSplit, DatasetType


ZENODO_RECORD_ID = "8224111"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Known datasets in the bundle
KNOWN_DATASETS = [
    "Abt-Buy",
    "Amazon-Google",
    "DBLP-ACM",
    "DBLP-Scholar",
    "iTunes-Amazon",
    "Walmart-Amazon",
    "BeerAdvo-RateBeer",
    "Fodors-Zagats",
    "Company",
]


class ZenodoProvider(DatasetProvider):
    """
    Provider for Zenodo 13 established ER datasets.
    
    Handles downloading, extracting, and loading classic ER benchmarks
    with clean and dirty variants.
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        variant: str = "clean",
        auto_download: bool = True,
        **kwargs
    ):
        """
        Initialize Zenodo dataset provider.
        
        Args:
            data_dir: Root directory for Zenodo datasets
            dataset_name: Name of dataset (e.g., "Abt-Buy", "Amazon-Google")
            variant: "clean" or "dirty"
            auto_download: Whether to auto-download if not present
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, **kwargs)
        self.dataset_name = dataset_name
        self.variant = variant
        self.auto_download = auto_download
        
        # Normalize dataset name
        self.dataset_name_normalized = dataset_name.lower().replace("-", "_")
        self.dataset_dir = os.path.join(
            data_dir, "zenodo", self.dataset_name_normalized, variant
        )
        
        if auto_download and not self._is_downloaded():
            self.download()
    
    def _is_downloaded(self) -> bool:
        """Check if dataset is already downloaded."""
        required_files = ["tableA.csv", "tableB.csv", "train.csv"]
        return all(
            os.path.exists(os.path.join(self.dataset_dir, f))
            for f in required_files
        )
    
    def download(self):
        """
        Download dataset from Zenodo if not already present.
        """
        if self._is_downloaded():
            print(f"{self.dataset_name} ({self.variant}) already downloaded.")
            return
        
        print(f"Downloading {self.dataset_name} ({self.variant}) from Zenodo...")
        
        # Create directory
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Fetch metadata
        try:
            response = requests.get(ZENODO_API_URL)
            response.raise_for_status()
            metadata = response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch Zenodo metadata: {e}")
        
        # Find the right file
        files = metadata.get("files", [])
        target_filename = f"{self.dataset_name}_{self.variant}.zip"
        
        file_info = None
        for f in files:
            if target_filename.lower() in f.get("key", "").lower():
                file_info = f
                break
        
        if file_info is None:
            # Try to find any matching file
            for f in files:
                key = f.get("key", "").lower()
                if self.dataset_name.lower().replace("-", "") in key:
                    if self.variant in key or (self.variant == "clean" and "dirty" not in key):
                        file_info = f
                        break
        
        if file_info is None:
            raise RuntimeError(
                f"Could not find {target_filename} in Zenodo record {ZENODO_RECORD_ID}"
            )
        
        # Download file
        download_url = file_info.get("links", {}).get("self")
        file_size = file_info.get("size", 0)
        
        zip_path = os.path.join(self.dataset_dir, "download.zip")
        
        print(f"Downloading {file_size / 1024 / 1024:.1f} MB...")
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            if file_size > 0:
                with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_dir)
        
        # Clean up zip
        os.remove(zip_path)
        
        print(f"Download complete: {self.dataset_dir}")
    
    def load(self) -> DatasetSplit:
        """
        Load Zenodo dataset splits.
        
        Returns:
            DatasetSplit with train/val/test pairs and entity tables
        """
        if not self._is_downloaded():
            raise RuntimeError(
                f"Dataset not downloaded. Call download() or set auto_download=True"
            )
        
        # Load entity tables
        left_table = pd.read_csv(
            os.path.join(self.dataset_dir, "tableA.csv"),
            dtype=str,
            keep_default_na=False
        )
        right_table = pd.read_csv(
            os.path.join(self.dataset_dir, "tableB.csv"),
            dtype=str,
            keep_default_na=False
        )
        
        # Standardize ID column
        if "id" not in left_table.columns and len(left_table.columns) > 0:
            # Assume first column is ID
            left_table = left_table.rename(columns={left_table.columns[0]: "id"})
        if "id" not in right_table.columns and len(right_table.columns) > 0:
            right_table = right_table.rename(columns={right_table.columns[0]: "id"})
        
        # Create text representation for each record
        if "text" not in left_table.columns:
            left_table["text"] = self._create_text_representation(left_table)
        if "text" not in right_table.columns:
            right_table["text"] = self._create_text_representation(right_table)
        
        # Load pair splits
        train_pairs = self._load_pairs("train.csv")
        val_pairs = self._load_pairs("valid.csv") if os.path.exists(
            os.path.join(self.dataset_dir, "valid.csv")
        ) else pd.DataFrame(columns=["ltable_id", "rtable_id", "label"])
        test_pairs = self._load_pairs("test.csv")
        
        metadata = {
            "name": self.dataset_name,
            "variant": self.variant,
            "type": "two_table",
            "n_left_entities": len(left_table),
            "n_right_entities": len(right_table),
        }
        
        return DatasetSplit(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            left_table=left_table,
            right_table=right_table,
            metadata=metadata,
        )
    
    def _load_pairs(self, filename: str) -> pd.DataFrame:
        """Load pair CSV and standardize format."""
        filepath = os.path.join(self.dataset_dir, filename)
        
        if not os.path.exists(filepath):
            return pd.DataFrame(columns=["ltable_id", "rtable_id", "label"])
        
        pairs = pd.read_csv(filepath, dtype=str, keep_default_na=False)
        
        # Standardize column names
        col_map = {}
        for col in pairs.columns:
            col_lower = col.lower()
            if "ltable" in col_lower or "left" in col_lower or col_lower == "id_a":
                col_map[col] = "ltable_id"
            elif "rtable" in col_lower or "right" in col_lower or col_lower == "id_b":
                col_map[col] = "rtable_id"
            elif "label" in col_lower or "match" in col_lower:
                col_map[col] = "label"
        
        pairs = pairs.rename(columns=col_map)
        
        # Ensure required columns exist
        for col in ["ltable_id", "rtable_id", "label"]:
            if col not in pairs.columns:
                pairs[col] = ""
        
        # Convert label to binary
        pairs["label"] = pd.to_numeric(pairs["label"], errors="coerce").fillna(0).astype(int)
        
        return pairs[["ltable_id", "rtable_id", "label"]]
    
    def _create_text_representation(self, table: pd.DataFrame) -> pd.Series:
        """Create text representation by concatenating non-ID columns."""
        text_cols = [col for col in table.columns if col.lower() not in ["id", "text"]]
        
        if not text_cols:
            return pd.Series([""] * len(table))
        
        texts = []
        for _, row in table.iterrows():
            parts = [str(row[col]) for col in text_cols if pd.notna(row[col]) and str(row[col]).strip()]
            texts.append(" ".join(parts))
        
        return pd.Series(texts)
    
    def get_ground_truth(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get ground truth labels.
        
        Returns:
            Tuple of (pair_labels, cluster_labels)
        """
        split = self.load()
        
        # Combine all pair splits for complete ground truth
        all_pairs = pd.concat([
            split.train_pairs,
            split.val_pairs,
            split.test_pairs
        ], ignore_index=True)
        
        # Cluster labels not directly available in two-table setting
        # Would need to compute connected components from pairs
        cluster_labels = pd.DataFrame(columns=["id", "cluster_id"])
        
        return all_pairs, cluster_labels
    
    def get_dataset_type(self) -> DatasetType:
        """Return dataset type."""
        return DatasetType.TWO_TABLE


def list_available_datasets() -> List[str]:
    """
    List all available datasets in the Zenodo bundle.
    
    Returns:
        List of dataset names
    """
    return KNOWN_DATASETS.copy()

