"""Base clusterer interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import numpy as np

from ..core.types import Pair, ClusterResult


class BaseClusterer(ABC):
    """
    Abstract base class for entity clustering.
    
    Clusters group matched records into coherent entities.
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
    
    @abstractmethod
    def cluster(
        self,
        edges: List[Tuple[int, int, float]],
        all_record_ids: List[int]
    ) -> ClusterResult:
        """
        Cluster records based on pairwise scores.
        
        Args:
            edges: List of (id1, id2, score) for positive pairs.
            all_record_ids: All record IDs to cluster.
            
        Returns:
            ClusterResult with clusters and labels.
        """
        raise NotImplementedError

