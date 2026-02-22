"""Geographic hierarchy blocking for ICE-ID."""

from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict
from .base import BlockingStrategy


class GeoHierarchyBlocking(BlockingStrategy):
    """
    Block records within the same geographic hierarchy.
    
    For ICE-ID: county → district → parish → farm hierarchy.
    Only generates pairs within the same geographic unit.
    """
    
    def __init__(
        self,
        hierarchy_levels: List[str] = None,
        max_pairs_per_block: int = 10000,
    ):
        """
        Initialize geographic hierarchy blocking.
        
        Args:
            hierarchy_levels: Geographic levels (coarsest to finest)
            max_pairs_per_block: Maximum pairs per geographic block
        """
        self.hierarchy_levels = hierarchy_levels or [
            "county", "district", "parish", "farm"
        ]
        self.max_pairs_per_block = max_pairs_per_block
    
    def generate_candidates(self, records: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Generate candidate pairs within geographic blocks.
        
        Args:
            records: DataFrame with id and geographic fields
            
        Returns:
            List of candidate pairs
        """
        # Try each hierarchy level from finest to coarsest
        for level in reversed(self.hierarchy_levels):
            if level in records.columns:
                # Use this level for blocking
                return self._block_by_field(records, level)
        
        # Fallback: use any available geographic field
        for col in records.columns:
            if any(geo in col.lower() for geo in ["county", "district", "parish", "farm", "location"]):
                return self._block_by_field(records, col)
        
        # No geographic fields found - return empty
        return []
    
    def _block_by_field(
        self,
        records: pd.DataFrame,
        field: str
    ) -> List[Tuple[int, int]]:
        """Block records by a single geographic field."""
        # Build inverted index: geo_value -> list of record IDs
        geo_index: Dict = defaultdict(list)
        
        for _, row in records.iterrows():
            rid = int(row["id"])
            geo_value = str(row.get(field, "")).strip()
            
            if geo_value and geo_value != "nan":
                geo_index[geo_value].append(rid)
        
        # Generate pairs within each geographic block
        pairs_set = set()
        
        for geo_value, ids in geo_index.items():
            if len(ids) < 2:
                continue
            
            # Skip very large blocks
            if len(ids) * (len(ids) - 1) // 2 > self.max_pairs_per_block:
                # For large blocks, subsample
                import random
                random.shuffle(ids)
                ids = ids[:100]  # Limit to 100 records per block
            
            # Generate pairs within block
            for i, id1 in enumerate(ids):
                for id2 in ids[i+1:]:
                    pair = (min(id1, id2), max(id1, id2))
                    pairs_set.add(pair)
        
        return list(pairs_set)

