"""
Deterministic rules-based entity resolution.

Hand-crafted rules for entity matching, useful as interpretable baseline.
"""

from typing import Dict, List, Set, Optional, Any, Callable
import numpy as np
import pandas as pd

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


class RulesModel(BaseModel):
    """
    Rule-based entity matcher.
    
    Applies a set of configurable rules to determine matches.
    Each rule contributes a score; final score is weighted average.
    """
    
    def __init__(
        self,
        rules: List[Dict[str, Any]] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize rules model.
        
        Args:
            rules: List of rule specifications, each with:
                - field: Field name to compare
                - type: "exact", "prefix", "soundex", "numeric_within"
                - weight: Rule weight (default 1.0)
                - params: Additional parameters
            threshold: Match decision threshold.
        """
        super().__init__("rules", **kwargs)
        self.rules = rules or self._default_rules()
        self.threshold = threshold
        self.weights = [r.get("weight", 1.0) for r in self.rules]
        self.total_weight = sum(self.weights)
    
    def _default_rules(self) -> List[Dict[str, Any]]:
        """Default rules for ICE-ID."""
        return [
            {"field": "nafn_norm", "type": "exact", "weight": 3.0},
            {"field": "first_name", "type": "exact", "weight": 2.0},
            {"field": "patronym", "type": "exact", "weight": 1.5},
            {"field": "surname", "type": "exact", "weight": 1.5},
            {"field": "birthyear", "type": "numeric_within", "weight": 2.0, "params": {"max_diff": 5}},
            {"field": "sex", "type": "exact", "weight": 1.0},
            {"field": "parish", "type": "exact", "weight": 0.5},
            {"field": "county", "type": "exact", "weight": 0.5},
        ]
    
    def _apply_rule(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any],
        rule: Dict[str, Any]
    ) -> float:
        """Apply a single rule, return score in [0, 1]."""
        field = rule["field"]
        rule_type = rule["type"]
        params = rule.get("params", {})
        
        v1 = rec1.get(field, "")
        v2 = rec2.get(field, "")
        
        if v1 is None or v2 is None:
            return 0.5
        
        v1 = str(v1).strip().lower()
        v2 = str(v2).strip().lower()
        
        if not v1 or not v2:
            return 0.5
        
        if rule_type == "exact":
            return 1.0 if v1 == v2 else 0.0
        
        elif rule_type == "prefix":
            min_len = params.get("min_len", 3)
            if len(v1) >= min_len and len(v2) >= min_len:
                return 1.0 if v1[:min_len] == v2[:min_len] else 0.0
            return 0.5
        
        elif rule_type == "numeric_within":
            max_diff = params.get("max_diff", 5)
            try:
                n1 = float(v1)
                n2 = float(v2)
                diff = abs(n1 - n2)
                if diff <= max_diff:
                    return 1.0 - (diff / (max_diff + 1))
                return 0.0
            except ValueError:
                return 0.5
        
        elif rule_type == "soundex":
            return 1.0 if self._soundex(v1) == self._soundex(v2) else 0.0
        
        elif rule_type == "jaccard":
            tokens1 = set(v1.split())
            tokens2 = set(v2.split())
            if not tokens1 or not tokens2:
                return 0.5
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0.5
        
        return 0.5
    
    def _soundex(self, s: str) -> str:
        """Simple Soundex implementation."""
        if not s:
            return ""
        
        s = s.upper()
        result = s[0]
        
        mapping = {
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'
        }
        
        for char in s[1:]:
            code = mapping.get(char, '0')
            if code != '0' and code != result[-1]:
                result += code
            if len(result) >= 4:
                break
        
        return result.ljust(4, '0')
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """Rules model doesn't need training."""
        self._is_fitted = True
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """Score pairs by applying all rules."""
        scores = []
        
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            rule_scores = []
            for rule, weight in zip(self.rules, self.weights):
                rule_score = self._apply_rule(rec1, rec2, rule)
                rule_scores.append(rule_score * weight)
            
            if self.total_weight > 0:
                final_score = sum(rule_scores) / self.total_weight
            else:
                final_score = 0.5
            
            scores.append(final_score)
        
        return np.array(scores)


get_registry("models").register("rules", RulesModel)

