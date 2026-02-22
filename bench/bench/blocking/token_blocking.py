"""Token-based blocking strategy."""

from typing import List, Set
from collections import defaultdict

from .base import BaseBlocker
from ..core.types import DatasetSplit, CandidatePairs
from ..core.registry import get_registry


class TokenBlocker(BaseBlocker):
    """
    Token blocking: group records by shared tokens.
    
    Records sharing at least one token in specified fields
    are placed in the same block.
    """
    
    def __init__(
        self,
        fields: List[str],
        min_token_length: int = 2,
        max_block_size: int = 1000,
        **kwargs
    ):
        """
        Initialize token blocker.
        
        Args:
            fields: Fields to extract tokens from.
            min_token_length: Minimum token length to consider.
            max_block_size: Maximum records per block (prune large blocks).
        """
        super().__init__("token_blocking", **kwargs)
        self.fields = fields
        self.min_token_length = min_token_length
        self.max_block_size = max_block_size
    
    def _get_tokens(self, record: dict) -> Set[str]:
        """Extract tokens from record fields."""
        tokens = set()
        for field in self.fields:
            value = str(record.get(field, "")).lower()
            for token in value.split():
                if len(token) >= self.min_token_length:
                    tokens.add(token)
        return tokens
    
    def block(
        self,
        dataset: DatasetSplit,
        record_ids: List[int] = None
    ) -> CandidatePairs:
        """Generate pairs from token blocks."""
        records = dataset.get_all_records()
        
        if record_ids is not None:
            records = records[records["id"].isin(record_ids)]
        
        blocks = defaultdict(list)
        
        for _, row in records.iterrows():
            record_id = int(row["id"])
            tokens = self._get_tokens(row.to_dict())
            for token in tokens:
                blocks[token].append(record_id)
        
        candidate_set = set()
        
        for token, members in blocks.items():
            if len(members) > self.max_block_size:
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


get_registry("blockers").register("token_blocking", TokenBlocker)

