"""
Base blocker interface for candidate pair generation.

Blocking reduces the quadratic comparison space by generating
only promising candidate pairs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set, Tuple
import pandas as pd

from ..core.types import DatasetSplit, CandidatePairs, Pair


class BaseBlocker(ABC):
    """
    Abstract base class for blocking strategies.
    
    Blockers generate candidate pairs from a dataset,
    reducing the O(n²) comparison space.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize blocker.
        
        Args:
            name: Blocker identifier.
            **kwargs: Blocker-specific parameters.
        """
        self.name = name
        self.params = kwargs
    
    @abstractmethod
    def block(
        self,
        dataset: DatasetSplit,
        record_ids: List[int] = None
    ) -> CandidatePairs:
        """
        Generate candidate pairs.
        
        Args:
            dataset: Dataset split containing records.
            record_ids: Optional subset of record IDs to consider.
            
        Returns:
            CandidatePairs with generated pairs and metadata.
        """
        raise NotImplementedError
    
    def evaluate(
        self,
        candidate_pairs: CandidatePairs,
        ground_truth_pairs: List[Pair]
    ) -> Dict[str, float]:
        """
        Evaluate blocking performance.
        
        Args:
            candidate_pairs: Generated candidate pairs.
            ground_truth_pairs: True positive pairs.
            
        Returns:
            Dict with metrics: recall, reduction_ratio, etc.
        """
        candidates = set(candidate_pairs.pairs)
        candidates.update((b, a) for a, b in candidate_pairs.pairs)
        
        truth = set(ground_truth_pairs)
        truth.update((b, a) for a, b in ground_truth_pairs)
        
        true_positives = len(candidates & truth)
        
        recall = true_positives / len(truth) if truth else 1.0
        
        n_records = len(candidate_pairs.metadata.get("record_ids", []))
        max_pairs = n_records * (n_records - 1) // 2 if n_records > 1 else 1
        
        reduction_ratio = 1.0 - (len(candidate_pairs.pairs) / max_pairs) if max_pairs > 0 else 0.0
        
        return {
            "blocking_recall": recall,
            "reduction_ratio": reduction_ratio,
            "n_candidates": len(candidate_pairs.pairs),
            "n_ground_truth": len(ground_truth_pairs),
            "n_true_positives": true_positives,
        }


class TrivialAllPairsBlocker(BaseBlocker):
    """Generate all possible pairs (O(n²)). Only for small datasets."""
    
    def __init__(self, **kwargs):
        super().__init__("trivial_allpairs", **kwargs)
    
    def block(
        self,
        dataset: DatasetSplit,
        record_ids: List[int] = None
    ) -> CandidatePairs:
        """Generate all pairs."""
        if record_ids is None:
            records = dataset.get_all_records()
            record_ids = records["id"].tolist()
        
        pairs = []
        for i in range(len(record_ids)):
            for j in range(i + 1, len(record_ids)):
                pairs.append((record_ids[i], record_ids[j]))
        
        return CandidatePairs(
            pairs=pairs,
            metadata={"record_ids": record_ids, "blocker": self.name}
        )

