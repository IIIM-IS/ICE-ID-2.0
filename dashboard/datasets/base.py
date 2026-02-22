"""
Base classes for dataset providers.

Defines the interface that all dataset providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd


class DatasetType(Enum):
    """Type of entity resolution task."""
    TWO_TABLE = "two_table"  # Link two separate tables (classic ER)
    DEDUP = "dedup"  # Deduplicate single table (ICE-ID style)
    

@dataclass
class DatasetSplit:
    """
    Container for train/validation/test splits.
    
    Attributes:
        train_pairs: Training pairs DataFrame with columns [ltable_id, rtable_id, label]
        val_pairs: Validation pairs DataFrame
        test_pairs: Test pairs DataFrame
        left_table: Left entity table (or all entities for dedup)
        right_table: Right entity table (None for dedup)
        metadata: Optional metadata dictionary
    """
    train_pairs: pd.DataFrame
    val_pairs: pd.DataFrame
    test_pairs: pd.DataFrame
    left_table: pd.DataFrame
    right_table: Optional[pd.DataFrame] = None
    metadata: Optional[Dict] = None


class DatasetProvider(ABC):
    """
    Abstract base class for dataset providers.
    
    All dataset providers must implement this interface to ensure
    compatibility with the benchmark framework.
    """
    
    def __init__(self, data_dir: str, **kwargs):
        """
        Initialize dataset provider.
        
        Args:
            data_dir: Root directory for dataset files
            **kwargs: Additional provider-specific arguments
        """
        self.data_dir = data_dir
        self.kwargs = kwargs
    
    @abstractmethod
    def load(self) -> DatasetSplit:
        """
        Load and return dataset splits.
        
        Returns:
            DatasetSplit containing train/val/test pairs and entity tables
        """
        pass
    
    @abstractmethod
    def get_ground_truth(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get ground truth labels.
        
        Returns:
            Tuple of (pair_labels, cluster_labels)
            - pair_labels: DataFrame with [id1, id2, label]
            - cluster_labels: DataFrame with [id, cluster_id]
        """
        pass
    
    @abstractmethod
    def get_dataset_type(self) -> DatasetType:
        """
        Return the type of ER task.
        
        Returns:
            DatasetType enum value
        """
        pass
    
    def get_statistics(self) -> Dict:
        """
        Compute and return dataset statistics.
        
        Returns:
            Dictionary with dataset statistics (entities, pairs, etc.)
        """
        split = self.load()
        stats = {
            "n_train_pairs": len(split.train_pairs),
            "n_val_pairs": len(split.val_pairs),
            "n_test_pairs": len(split.test_pairs),
            "n_left_entities": len(split.left_table),
        }
        
        if split.right_table is not None:
            stats["n_right_entities"] = len(split.right_table)
        
        if len(split.train_pairs) > 0:
            stats["train_pos_ratio"] = split.train_pairs["label"].mean()
        
        return stats
    
    def as_two_table(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return dataset in two-table format.
        
        For dedup tasks, splits entities into two tables.
        
        Returns:
            Tuple of (left_table, right_table)
        """
        split = self.load()
        if split.right_table is not None:
            return split.left_table, split.right_table
        else:
            # Split dedup table into two halves
            n = len(split.left_table)
            mid = n // 2
            return split.left_table.iloc[:mid], split.left_table.iloc[mid:]
    
    def as_dedup_table(self) -> pd.DataFrame:
        """
        Return dataset in deduplication format.
        
        For two-table tasks, combines both tables.
        
        Returns:
            Single entity table
        """
        split = self.load()
        if split.right_table is None:
            return split.left_table
        else:
            # Combine both tables
            return pd.concat([split.left_table, split.right_table], ignore_index=True)

