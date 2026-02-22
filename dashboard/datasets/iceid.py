"""
ICE-ID dataset provider.

Wraps existing ICE-ID functionality into the DatasetProvider interface.
"""

import os
from typing import Dict, Tuple
import pandas as pd

from .base import DatasetProvider, DatasetSplit, DatasetType


class ICEIDProvider(DatasetProvider):
    """Provider for the ICE-ID historical census dataset."""
    
    def __init__(
        self,
        data_dir: str,
        people_csv: str = None,
        labels_csv: str = None,
        **kwargs
    ):
        """
        Initialize ICE-ID provider.
        
        Args:
            data_dir: Root directory containing ICE-ID data
            people_csv: Path to people.csv (overrides default)
            labels_csv: Path to labels CSV (overrides default)
            **kwargs: Additional arguments
        """
        super().__init__(data_dir, **kwargs)
        
        self.people_csv = people_csv or os.path.join(data_dir, "people.csv")
        self.labels_csv = labels_csv or os.path.join(
            data_dir, "manntol_einstaklingar_new.csv"
        )
    
    def load(self) -> DatasetSplit:
        """
        Load ICE-ID dataset splits.
        
        Returns:
            DatasetSplit with temporal train/val/test splits
        """
        # Load people data
        people = pd.read_csv(
            self.people_csv,
            dtype=str,
            keep_default_na=False,
            low_memory=False
        )
        
        # Load labels for ground truth
        labels = pd.read_csv(
            self.labels_csv,
            dtype=str,
            keep_default_na=False,
            low_memory=False
        )
        
        # Convert IDs to numeric
        for col in ["id", "person"]:
            if col in people.columns:
                people[col] = pd.to_numeric(people[col], errors="coerce")
        
        people = people.dropna(subset=["id"]).copy()
        people["id"] = people["id"].astype(int)
        
        # Parse census year (heimild)
        if "heimild" in people.columns:
            people["heimild"] = pd.to_numeric(people["heimild"], errors="coerce")
        
        # Create temporal splits based on paper specification:
        # Train: pre-1870, Val: 1870-1900, Test: 1901-1920
        train_mask = people["heimild"] < 1870
        val_mask = (people["heimild"] >= 1870) & (people["heimild"] <= 1900)
        test_mask = people["heimild"] > 1900
        
        # For now, return empty pair dataframes
        # Actual pair generation will be done by the experiment runner
        empty_pairs = pd.DataFrame(columns=["ltable_id", "rtable_id", "label"])
        
        metadata = {
            "name": "ICE-ID",
            "type": "deduplication",
            "temporal_split": True,
            "n_train_records": train_mask.sum(),
            "n_val_records": val_mask.sum(),
            "n_test_records": test_mask.sum(),
        }
        
        return DatasetSplit(
            train_pairs=empty_pairs,
            val_pairs=empty_pairs,
            test_pairs=empty_pairs,
            left_table=people,
            right_table=None,
            metadata=metadata,
        )
    
    def get_ground_truth(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get ground truth cluster labels.
        
        Returns:
            Tuple of (pair_labels, cluster_labels)
        """
        people = pd.read_csv(
            self.people_csv,
            dtype=str,
            keep_default_na=False,
            low_memory=False
        )
        
        for col in ["id", "person"]:
            if col in people.columns:
                people[col] = pd.to_numeric(people[col], errors="coerce")
        
        people = people.dropna(subset=["id"]).copy()
        
        cluster_labels = people[["id", "person"]].copy()
        cluster_labels.columns = ["id", "cluster_id"]
        
        # Empty pair labels (will be generated on demand)
        pair_labels = pd.DataFrame(columns=["id1", "id2", "label"])
        
        return pair_labels, cluster_labels
    
    def get_dataset_type(self) -> DatasetType:
        """Return dataset type."""
        return DatasetType.DEDUP

