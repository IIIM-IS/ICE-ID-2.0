"""Geographic hierarchy blocking for ICE-ID."""

from typing import List, Tuple
from collections import defaultdict

from .base import BaseBlocker
from ..core.types import DatasetSplit, CandidatePairs
from ..core.registry import get_registry


class GeoHierarchyBlocker(BaseBlocker):
    """
    Geographic hierarchy blocking for ICE-ID.
    
    Groups records by hierarchical geographic regions:
    county → district → parish → farm
    """
    
    def __init__(
        self,
        levels: List[str] = None,
        max_level_diff: int = 2,
        **kwargs
    ):
        """
        Initialize geo hierarchy blocker.
        
        Args:
            levels: Geographic levels from coarsest to finest.
                    Default: ["county", "district", "parish", "farm"]
            max_level_diff: Allow matching across N level differences.
        """
        super().__init__("geo_hierarchy_blocking", **kwargs)
        self.levels = levels or ["county", "district", "parish", "farm"]
        self.max_level_diff = max_level_diff
    
    def _get_geo_key(self, record: dict) -> Tuple[str, ...]:
        """Get hierarchical geographic key."""
        key_parts = []
        for level in self.levels:
            value = str(record.get(level, "")).strip().lower()
            if value:
                key_parts.append(value)
            else:
                break
        return tuple(key_parts)
    
    def block(
        self,
        dataset: DatasetSplit,
        record_ids: List[int] = None
    ) -> CandidatePairs:
        """Generate pairs within geographic blocks."""
        records = dataset.get_all_records()
        
        if record_ids is not None:
            records = records[records["id"].isin(record_ids)]
        
        blocks = defaultdict(list)
        
        for _, row in records.iterrows():
            record_id = int(row["id"])
            geo_key = self._get_geo_key(row.to_dict())
            
            for prefix_len in range(1, len(geo_key) + 1):
                prefix = geo_key[:prefix_len]
                blocks[prefix].append(record_id)
        
        candidate_set = set()
        
        for geo_key, members in blocks.items():
            if len(members) > 5000:
                continue
            
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    id1, id2 = sorted((members[i], members[j]))
                    candidate_set.add((id1, id2))
        
        return CandidatePairs(
            pairs=sorted(list(candidate_set)),
            metadata={
                "record_ids": records["id"].tolist(),
                "n_blocks": len(blocks),
                "blocker": self.name,
            }
        )


get_registry("blockers").register("geo_hierarchy_blocking", GeoHierarchyBlocker)

