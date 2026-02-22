"""Token-based blocking using shared tokens in fields."""

from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict
from .base import BlockingStrategy


class TokenBlocking(BlockingStrategy):
    """
    Block records that share at least one token in specified fields.
    
    Tokens are extracted by splitting fields on whitespace.
    """
    
    def __init__(
        self,
        blocking_fields: List[str] = None,
        min_token_length: int = 2,
        max_pairs_per_token: int = 10000,
    ):
        """
        Initialize token blocking.
        
        Args:
            blocking_fields: Fields to extract tokens from
            min_token_length: Minimum token length to consider
            max_pairs_per_token: Maximum pairs per token block
        """
        self.blocking_fields = blocking_fields or ["text", "name"]
        self.min_token_length = min_token_length
        self.max_pairs_per_token = max_pairs_per_token
    
    def _extract_tokens(self, value: str) -> Set[str]:
        """Extract tokens from a field value."""
        if not isinstance(value, str) or not value.strip():
            return set()
        
        tokens = value.lower().split()
        return {
            t for t in tokens
            if len(t) >= self.min_token_length and t.isalnum()
        }
    
    def generate_candidates(self, records: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Generate candidate pairs using token blocking.
        
        Args:
            records: DataFrame with id and blocking fields
            
        Returns:
            List of candidate pairs
        """
        # Build inverted index: token -> list of record IDs
        token_index: Dict[str, List[int]] = defaultdict(list)
        
        for _, row in records.iterrows():
            rid = int(row["id"])
            tokens = set()
            
            for field in self.blocking_fields:
                if field in row:
                    value = row[field]
                    tokens.update(self._extract_tokens(str(value)))
            
            for token in tokens:
                token_index[token].append(rid)
        
        # Generate pairs from blocks
        pairs_set = set()
        
        for token, ids in token_index.items():
            if len(ids) < 2:
                continue
            
            # Skip very large blocks
            if len(ids) * (len(ids) - 1) // 2 > self.max_pairs_per_token:
                continue
            
            # Generate pairs within block
            for i, id1 in enumerate(ids):
                for id2 in ids[i+1:]:
                    pair = (min(id1, id2), max(id1, id2))
                    pairs_set.add(pair)
        
        return list(pairs_set)

