"""
Base blocking strategy interface.

All blocking strategies must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import pandas as pd


class BlockingStrategy(ABC):
    """Abstract base class for blocking strategies."""
    
    @abstractmethod
    def generate_candidates(self, records: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Generate candidate pairs from records.
        
        Args:
            records: DataFrame with record features
            
        Returns:
            List of (id1, id2) candidate pairs
        """
        pass
    
    def get_metrics(
        self,
        candidates: List[Tuple[int, int]],
        ground_truth: pd.DataFrame = None,
    ) -> Dict:
        """
        Compute blocking metrics.
        
        Args:
            candidates: Generated candidate pairs
            ground_truth: Optional ground truth pairs for recall computation
            
        Returns:
            Dictionary with blocking metrics
        """
        metrics = {
            "n_candidates": len(candidates),
        }
        
        if ground_truth is not None and len(ground_truth) > 0:
            # Compute recall: fraction of true matches captured
            candidate_set = set(candidates)
            true_pairs = set()
            
            for _, row in ground_truth.iterrows():
                id1, id2 = int(row["id1"]), int(row["id2"])
                pair = (min(id1, id2), max(id1, id2))
                true_pairs.add(pair)
            
            captured = len(candidate_set & true_pairs)
            recall = captured / len(true_pairs) if len(true_pairs) > 0 else 0.0
            
            metrics["pairs_captured"] = captured
            metrics["total_true_pairs"] = len(true_pairs)
            metrics["recall"] = recall
        
        return metrics

