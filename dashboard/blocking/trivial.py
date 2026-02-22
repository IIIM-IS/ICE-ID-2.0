"""Trivial all-pairs blocking for small datasets."""

from typing import List, Tuple
import pandas as pd
from .base import BlockingStrategy


class TrivialBlocking(BlockingStrategy):
    """
    Generate all possible pairs (Cartesian product).
    
    Only suitable for very small datasets due to quadratic complexity.
    """
    
    def __init__(self, max_records: int = 1000):
        """
        Initialize trivial blocking.
        
        Args:
            max_records: Maximum number of records to allow (safety check)
        """
        self.max_records = max_records
    
    def generate_candidates(self, records: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Generate all possible pairs.
        
        Args:
            records: DataFrame with id column
            
        Returns:
            List of all (id1, id2) pairs where id1 < id2
        """
        ids = records["id"].tolist()
        
        if len(ids) > self.max_records:
            raise ValueError(
                f"Too many records ({len(ids)}) for trivial blocking. "
                f"Maximum allowed: {self.max_records}"
            )
        
        pairs = []
        for i, id1 in enumerate(ids):
            for id2 in ids[i+1:]:
                pairs.append((int(id1), int(id2)))
        
        return pairs

