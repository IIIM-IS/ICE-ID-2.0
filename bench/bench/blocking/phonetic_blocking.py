"""Phonetic blocking using Soundex/Metaphone."""

from typing import List, Set
from collections import defaultdict

from .base import BaseBlocker
from ..core.types import DatasetSplit, CandidatePairs
from ..core.registry import get_registry


class PhoneticBlocker(BaseBlocker):
    """
    Phonetic blocking: group records by phonetic codes.
    
    Uses Soundex or Double Metaphone to group records with
    similar-sounding names.
    """
    
    def __init__(
        self,
        fields: List[str],
        method: str = "soundex",
        **kwargs
    ):
        """
        Initialize phonetic blocker.
        
        Args:
            fields: Name fields to apply phonetic coding.
            method: "soundex" or "metaphone".
        """
        super().__init__("phonetic_blocking", **kwargs)
        self.fields = fields
        self.method = method
    
    def _soundex(self, s: str) -> str:
        """Compute Soundex code."""
        if not s:
            return ""
        
        s = "".join(c for c in s.upper() if c.isalpha())
        if not s:
            return ""
        
        result = s[0]
        
        mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        prev_code = mapping.get(s[0], '0')
        
        for char in s[1:]:
            code = mapping.get(char, '0')
            if code != '0' and code != prev_code:
                result += code
                prev_code = code
            if len(result) >= 4:
                break
        
        return result.ljust(4, '0')
    
    def _get_phonetic_codes(self, record: dict) -> Set[str]:
        """Get phonetic codes for record fields."""
        codes = set()
        for field in self.fields:
            value = str(record.get(field, "")).strip()
            for word in value.split():
                if self.method == "soundex":
                    code = self._soundex(word)
                    if code:
                        codes.add(code)
        return codes
    
    def block(
        self,
        dataset: DatasetSplit,
        record_ids: List[int] = None
    ) -> CandidatePairs:
        """Generate pairs from phonetic blocks."""
        records = dataset.get_all_records()
        
        if record_ids is not None:
            records = records[records["id"].isin(record_ids)]
        
        blocks = defaultdict(list)
        
        for _, row in records.iterrows():
            record_id = int(row["id"])
            codes = self._get_phonetic_codes(row.to_dict())
            for code in codes:
                blocks[code].append(record_id)
        
        candidate_set = set()
        
        for code, members in blocks.items():
            if len(members) > 1000:
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


get_registry("blockers").register("phonetic_blocking", PhoneticBlocker)

