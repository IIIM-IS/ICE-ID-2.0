"""
Non-Axiomatic Reasoning System (NARS) for Entity Resolution.

Implementation based on the paper specification (main_nars_paper.tex).
NARS uses pattern matching with truth values for entity matching.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, FrozenSet, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
import heapq


@dataclass(frozen=True)
class TruthValue:
    """
    Truth value in NARS with frequency and confidence.
    
    Attributes:
        f: Frequency (0 to 1) - proportion of positive evidence
        c: Confidence (0 to 1) - reliability of the frequency
    """
    f: float  # frequency
    c: float  # confidence
    
    def expectation(self) -> float:
        """
        Compute expectation: e = c(f - 0.5) + 0.5
        
        Returns:
            Expectation value in [0, 1]
        """
        return self.c * (self.f - 0.5) + 0.5
    
    @staticmethod
    def from_evidence(w_pos: float, w_neg: float, k: float = 1.0) -> TruthValue:
        """
        Create truth value from evidence counts.
        
        Args:
            w_pos: Positive evidence weight
            w_neg: Negative evidence weight
            k: Horizon constant (default 1.0)
            
        Returns:
            TruthValue instance
        """
        total = w_pos + w_neg
        if total == 0:
            return TruthValue(0.5, 0.0)
        
        f = w_pos / total
        c = total / (total + k)
        return TruthValue(f, c)
    
    @staticmethod
    def revision(tv1: TruthValue, tv2: TruthValue, k: float = 1.0) -> TruthValue:
        """
        Revise two truth values from independent sources.
        
        Args:
            tv1: First truth value
            tv2: Second truth value
            k: Horizon constant
            
        Returns:
            Revised truth value
        """
        # Convert to evidence
        w1_pos = tv1.f * tv1.c * k / (1 - tv1.c) if tv1.c < 1.0 else tv1.f * k * 10
        w1_neg = (1 - tv1.f) * tv1.c * k / (1 - tv1.c) if tv1.c < 1.0 else (1 - tv1.f) * k * 10
        
        w2_pos = tv2.f * tv2.c * k / (1 - tv2.c) if tv2.c < 1.0 else tv2.f * k * 10
        w2_neg = (1 - tv2.f) * tv2.c * k / (1 - tv2.c) if tv2.c < 1.0 else (1 - tv2.f) * k * 10
        
        # Combine evidence
        w_pos = w1_pos + w2_pos
        w_neg = w1_neg + w2_neg
        
        return TruthValue.from_evidence(w_pos, w_neg, k)


@dataclass(frozen=True)
class Pattern:
    """
    Pattern representing a set of judgments about a record pair.
    
    Attributes:
        judgments: Frozenset of judgment strings
        truth_value: Associated truth value
    """
    judgments: FrozenSet[str]
    truth_value: TruthValue
    
    def __len__(self) -> int:
        return len(self.judgments)
    
    def intersection(self, other: Pattern) -> FrozenSet[str]:
        """Get intersection of judgments."""
        return self.judgments & other.judgments
    
    def difference(self, other: Pattern) -> FrozenSet[str]:
        """Get difference of judgments."""
        return self.judgments - other.judgments
    
    def similarity(self, other: Pattern) -> float:
        """
        Compute similarity based on judgment overlap.
        
        Returns:
            Similarity score (Jaccard-like)
        """
        if len(self.judgments) == 0 and len(other.judgments) == 0:
            return 1.0
        
        intersection = len(self.intersection(other))
        longest = max(len(self.judgments), len(other.judgments))
        
        return intersection / longest if longest > 0 else 0.0


class PatternPool:
    """
    Pool of patterns sorted by expectation for efficient matching.
    
    Maintains patterns with their truth values and provides
    similarity-based retrieval.
    """
    
    def __init__(self, max_size: int = 10000, k: float = 1.0):
        """
        Initialize pattern pool.
        
        Args:
            max_size: Maximum number of patterns to store
            k: Horizon constant for truth value computations
        """
        self.max_size = max_size
        self.k = k
        self.patterns: List[Pattern] = []
        self.pattern_map: Dict[FrozenSet[str], TruthValue] = {}
    
    def add_pattern(self, judgments: Set[str], truth_value: TruthValue):
        """
        Add or update a pattern in the pool.
        
        Args:
            judgments: Set of judgment strings
            truth_value: Associated truth value
        """
        frozen_j = frozenset(judgments)
        
        if frozen_j in self.pattern_map:
            # Revise existing pattern
            existing_tv = self.pattern_map[frozen_j]
            new_tv = TruthValue.revision(existing_tv, truth_value, self.k)
            self.pattern_map[frozen_j] = new_tv
            
            # Update in list
            for i, p in enumerate(self.patterns):
                if p.judgments == frozen_j:
                    self.patterns[i] = Pattern(frozen_j, new_tv)
                    break
        else:
            # Add new pattern
            pattern = Pattern(frozen_j, truth_value)
            self.patterns.append(pattern)
            self.pattern_map[frozen_j] = truth_value
            
            # Keep pool size limited
            if len(self.patterns) > self.max_size:
                # Remove pattern with lowest expectation
                self.patterns.sort(key=lambda p: p.truth_value.expectation())
                removed = self.patterns.pop(0)
                del self.pattern_map[removed.judgments]
    
    def get_top_n(self, n: int, from_top: bool = True) -> List[Pattern]:
        """
        Get top N patterns by expectation.
        
        Args:
            n: Number of patterns to retrieve
            from_top: If True, get highest expectation; else lowest
            
        Returns:
            List of patterns
        """
        sorted_patterns = sorted(
            self.patterns,
            key=lambda p: p.truth_value.expectation(),
            reverse=from_top
        )
        return sorted_patterns[:n]
    
    def match_pattern(self, query_judgments: Set[str], n_patterns: int = 10) -> TruthValue:
        """
        Match a query pattern against the pool.
        
        Retrieves n_patterns/2 from high expectation (matches) and
        n_patterns/2 from low expectation (non-matches) to compute match score.
        
        Args:
            query_judgments: Set of judgment strings for query
            n_patterns: Number of patterns to use for matching
            
        Returns:
            Truth value representing match belief
        """
        n_half = n_patterns // 2
        
        # Get patterns with high and low expectations
        high_patterns = self.get_top_n(n_half, from_top=True)
        low_patterns = self.get_top_n(n_half, from_top=False)
        
        query_frozen = frozenset(query_judgments)
        match_tvs = []
        
        for patterns in [high_patterns, low_patterns]:
            for pattern in patterns:
                # Compute similarity
                intersection = len(query_frozen & pattern.judgments)
                longest = max(len(query_frozen), len(pattern.judgments))
                
                if longest == 0:
                    continue
                
                sim_f = intersection / longest
                sim_c = pattern.truth_value.c
                
                # Create similarity truth value
                sim_tv = TruthValue(sim_f, sim_c)
                
                # Revise with pattern's truth value
                revised = TruthValue.revision(sim_tv, pattern.truth_value, self.k)
                match_tvs.append(revised)
        
        if not match_tvs:
            return TruthValue(0.5, 0.0)
        
        # Combine all truth values via revision
        result = match_tvs[0]
        for tv in match_tvs[1:]:
            result = TruthValue.revision(result, tv, self.k)
        
        return result


def preprocess_iceid(row1: pd.Series, row2: pd.Series, age_disparity_threshold: int = 76) -> Set[str]:
    """
    Generate judgments for two ICE-ID records.
    
    Based on paper description (main_nars_paper.tex lines 402-418).
    
    Args:
        row1: First record
        row2: Second record
        age_disparity_threshold: Maximum age difference (default 76 from paper)
        
    Returns:
        Set of judgment strings
    """
    judgments = set()
    
    # Heimild (census year difference)
    try:
        h1 = int(row1.get("heimild", 0))
        h2 = int(row2.get("heimild", 0))
        diff = abs(h1 - h2)
        judgments.add(f"differ_in_{diff}_years")
    except (ValueError, TypeError):
        pass
    
    # Name comparisons
    for field in ["nafn_norm", "first_name", "patronym", "surname"]:
        v1 = str(row1.get(field, "")).strip().lower()
        v2 = str(row2.get(field, "")).strip().lower()
        
        if v1 and v2:
            if v1 == v2:
                judgments.add(f"same_{field}")
            else:
                judgments.add(f"different_{field}")
    
    # Birth year
    try:
        by1 = int(row1.get("birthyear", -1))
        by2 = int(row2.get("birthyear", -1))
        
        if by1 > 0 and by2 > 0:
            if by1 == by2:
                judgments.add("same_birthyear")
            elif abs(by1 - by2) <= age_disparity_threshold:
                judgments.add("birthyear_compatible_within_threshold")
            else:
                judgments.add("different_birthyear")
    except (ValueError, TypeError):
        pass
    
    # Sex
    s1 = str(row1.get("sex", "")).strip().lower()
    s2 = str(row2.get("sex", "")).strip().lower()
    if s1 and s2:
        if s1 == s2:
            judgments.add("same_sex")
        else:
            judgments.add("different_sex")
    
    # Status (add both values as per paper)
    st1 = str(row1.get("status", "")).strip()
    st2 = str(row2.get("status", "")).strip()
    if st1:
        judgments.add(f"status_is_{st1}")
    if st2:
        judgments.add(f"status_is_{st2}")
    
    # Marriage status
    ms1 = str(row1.get("marriagestatus", "")).strip().lower()
    ms2 = str(row2.get("marriagestatus", "")).strip().lower()
    if ms1 and ms2:
        if ms1 == ms2:
            judgments.add("same_marriagestatus")
        else:
            judgments.add("different_marriagestatus")
    
    # Location comparisons
    for field in ["farm", "county", "parish", "district"]:
        v1 = str(row1.get(field, "")).strip()
        v2 = str(row2.get(field, "")).strip()
        
        if v1 and v2:
            if v1 == v2:
                judgments.add(f"same_{field}")
            else:
                judgments.add(f"different_{field}")
    
    return judgments


class NARSMatcher:
    """
    NARS-based entity matcher for record linkage.
    """
    
    def __init__(
        self,
        pool_size: int = 10000,
        n_patterns_for_match: int = 10,
        k: float = 1.0,
        age_disparity_threshold: int = 76,
    ):
        """
        Initialize NARS matcher.
        
        Args:
            pool_size: Maximum patterns in pool
            n_patterns_for_match: Number of patterns to use for matching
            k: Horizon constant
            age_disparity_threshold: Max age difference for ICE-ID
        """
        self.pool = PatternPool(max_size=pool_size, k=k)
        self.n_patterns = n_patterns_for_match
        self.age_disparity_threshold = age_disparity_threshold
        self.threshold = 0.5  # Will be calibrated
    
    def fit(
        self,
        records: pd.DataFrame,
        pairs: pd.DataFrame,
        calibrate: bool = True,
    ):
        """
        Train NARS matcher on labeled pairs.
        
        Args:
            records: DataFrame with record features
            pairs: DataFrame with columns [id1, id2, label]
            calibrate: Whether to calibrate threshold
        """
        records_idx = records.set_index("id")
        
        # Learn patterns from pairs
        for _, row in pairs.iterrows():
            id1, id2, label = int(row["id1"]), int(row["id2"]), int(row["label"])
            
            if id1 not in records_idx.index or id2 not in records_idx.index:
                continue
            
            rec1 = records_idx.loc[id1]
            rec2 = records_idx.loc[id2]
            
            judgments = preprocess_iceid(rec1, rec2, self.age_disparity_threshold)
            
            # Create truth value based on label
            if label == 1:
                tv = TruthValue(1.0, 0.9)  # Match
            else:
                tv = TruthValue(0.0, 0.9)  # Non-match
            
            self.pool.add_pattern(judgments, tv)
        
        # Calibrate threshold if requested
        if calibrate and len(pairs) > 0:
            self._calibrate_threshold(records, pairs)
    
    def _calibrate_threshold(self, records: pd.DataFrame, pairs: pd.DataFrame):
        """
        Calibrate threshold using median method from paper.
        
        τ = (median(pos_scores) + median(neg_scores)) / 2
        """
        records_idx = records.set_index("id")
        
        pos_scores = []
        neg_scores = []
        
        for _, row in pairs.iterrows():
            id1, id2, label = int(row["id1"]), int(row["id2"]), int(row["label"])
            
            if id1 not in records_idx.index or id2 not in records_idx.index:
                continue
            
            rec1 = records_idx.loc[id1]
            rec2 = records_idx.loc[id2]
            
            score = self.score_pair(rec1, rec2)
            
            if label == 1:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        
        if pos_scores and neg_scores:
            median_pos = np.median(pos_scores)
            median_neg = np.median(neg_scores)
            self.threshold = (median_pos + median_neg) / 2
        elif pos_scores:
            self.threshold = np.median(pos_scores) * 0.9
        else:
            self.threshold = 0.5
    
    def score_pair(self, rec1: pd.Series, rec2: pd.Series) -> float:
        """
        Score a pair of records.
        
        Args:
            rec1: First record
            rec2: Second record
            
        Returns:
            Match score (expectation value)
        """
        judgments = preprocess_iceid(rec1, rec2, self.age_disparity_threshold)
        match_tv = self.pool.match_pattern(judgments, self.n_patterns)
        return match_tv.expectation()
    
    def predict(
        self,
        records: pd.DataFrame,
        pairs: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict matches for pairs.
        
        Args:
            records: DataFrame with record features
            pairs: DataFrame with columns [id1, id2]
            
        Returns:
            Binary predictions (1 = match, 0 = non-match)
        """
        scores = self.score_pairs(records, pairs)
        return (scores >= self.threshold).astype(int)
    
    def score_pairs(
        self,
        records: pd.DataFrame,
        pairs: pd.DataFrame,
    ) -> np.ndarray:
        """
        Score multiple pairs.
        
        Args:
            records: DataFrame with record features
            pairs: DataFrame with columns [id1, id2]
            
        Returns:
            Array of match scores
        """
        records_idx = records.set_index("id")
        scores = []
        
        for _, row in pairs.iterrows():
            id1, id2 = int(row["id1"]), int(row["id2"])
            
            if id1 not in records_idx.index or id2 not in records_idx.index:
                scores.append(0.0)
                continue
            
            rec1 = records_idx.loc[id1]
            rec2 = records_idx.loc[id2]
            
            score = self.score_pair(rec1, rec2)
            scores.append(score)
        
        return np.array(scores)

