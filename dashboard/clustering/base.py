"""Base clusterer interface."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd


class Clusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def cluster(
        self,
        edges: pd.DataFrame = None,
        scores: np.ndarray = None,
        record_ids: List[int] = None,
    ) -> List[int]:
        """
        Cluster records based on edges or scores.
        
        Args:
            edges: DataFrame with columns [id1, id2] (and optionally score/weight)
            scores: Pairwise similarity scores (alternative to edges)
            record_ids: List of all record IDs
            
        Returns:
            List of cluster assignments (parallel to record_ids)
        """
        pass

