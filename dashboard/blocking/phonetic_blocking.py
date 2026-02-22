"""Phonetic blocking using Soundex and DoubleMetaphone."""

from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict
from .base import BlockingStrategy

# Try to import phonetic libraries
try:
    import jellyfish
    HAS_JELLYFISH = True
except ImportError:
    HAS_JELLYFISH = False


class PhoneticBlocking(BlockingStrategy):
    """
    Block records with similar phonetic codes.
    
    Uses Soundex and/or DoubleMetaphone for name-based blocking.
    Particularly useful for ICE-ID patronymic names.
    """
    
    def __init__(
        self,
        name_fields: List[str] = None,
        method: str = "soundex",
        max_pairs_per_block: int = 10000,
    ):
        """
        Initialize phonetic blocking.
        
        Args:
            name_fields: Name fields to use for blocking
            method: "soundex" or "metaphone"
            max_pairs_per_block: Maximum pairs per block
        """
        if not HAS_JELLYFISH:
            raise ImportError(
                "jellyfish library required for phonetic blocking. "
                "Install with: pip install jellyfish"
            )
        
        self.name_fields = name_fields or ["name", "nafn_norm", "first_name", "patronym"]
        self.method = method
        self.max_pairs_per_block = max_pairs_per_block
    
    def _phonetic_code(self, name: str) -> str:
        """Get phonetic code for a name."""
        if not name or not isinstance(name, str):
            return ""
        
        name = name.strip()
        if not name:
            return ""
        
        if self.method == "soundex":
            try:
                return jellyfish.soundex(name)
            except Exception:
                return ""
        elif self.method == "metaphone":
            try:
                return jellyfish.metaphone(name)
            except Exception:
                return ""
        else:
            return name[:4].upper()  # Fallback: first 4 chars
    
    def generate_candidates(self, records: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Generate candidate pairs using phonetic blocking.
        
        Args:
            records: DataFrame with id and name fields
            
        Returns:
            List of candidate pairs
        """
        # Build inverted index: phonetic_code -> list of record IDs
        code_index: Dict[str, List[int]] = defaultdict(list)
        
        for _, row in records.iterrows():
            rid = int(row["id"])
            codes = set()
            
            for field in self.name_fields:
                if field in row:
                    name = str(row[field]).strip()
                    if name:
                        code = self._phonetic_code(name)
                        if code:
                            codes.add(code)
            
            for code in codes:
                code_index[code].append(rid)
        
        # Generate pairs from blocks
        pairs_set = set()
        
        for code, ids in code_index.items():
            if len(ids) < 2:
                continue
            
            # Skip very large blocks
            if len(ids) * (len(ids) - 1) // 2 > self.max_pairs_per_block:
                continue
            
            # Generate pairs within block
            for i, id1 in enumerate(ids):
                for id2 in ids[i+1:]:
                    pair = (min(id1, id2), max(id1, id2))
                    pairs_set.add(pair)
        
        return list(pairs_set)

