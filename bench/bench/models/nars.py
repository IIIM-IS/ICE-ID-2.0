"""
NARS (Non-Axiomatic Reasoning System) model adapter.

Implements entity resolution using NAL truth value semantics
with pattern-based evidence accumulation.

Reference: OpenNARS for Applications
https://github.com/opennars/OpenNARS-for-Applications
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, FrozenSet, Optional, Any
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from collections import defaultdict

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


@dataclass(frozen=True)
class TruthValue:
    """
    NAL Truth Value with frequency and confidence.
    
    Frequency (f): proportion of positive evidence
    Confidence (c): amount of evidence relative to max
    """
    f: float
    c: float
    
    @property
    def expectation(self) -> float:
        """Decision-making expectation: e = c * (f - 0.5) + 0.5"""
        return self.c * (self.f - 0.5) + 0.5
    
    @property
    def w_plus(self) -> float:
        """Positive evidence weight."""
        if self.c >= 0.9999:
            return 1e6 * self.f
        return self.f * self.c / (1 - self.c)
    
    @property
    def w_minus(self) -> float:
        """Negative evidence weight."""
        if self.c >= 0.9999:
            return 1e6 * (1 - self.f)
        return (1 - self.f) * self.c / (1 - self.c)
    
    @staticmethod
    def from_evidence(w_plus: float, w_minus: float, k: float = 1.0) -> 'TruthValue':
        """Create TruthValue from evidence counts."""
        total = w_plus + w_minus
        if total == 0:
            return TruthValue(0.5, 0.0)
        f = w_plus / total
        c = total / (total + k)
        return TruthValue(f, min(c, 0.9999))
    
    @staticmethod
    def revision(tv1: 'TruthValue', tv2: 'TruthValue', k: float = 1.0) -> 'TruthValue':
        """NAL revision rule for combining evidence."""
        w_plus = tv1.w_plus + tv2.w_plus
        w_minus = tv1.w_minus + tv2.w_minus
        return TruthValue.from_evidence(w_plus, w_minus, k)


class JudgmentStatistics:
    """
    Track statistics for each judgment type.
    
    For each judgment (e.g., "same_first_name"), track how often
    it appears in matching vs non-matching pairs.
    """
    
    def __init__(self, k: float = 1.0):
        """
        Initialize judgment statistics tracker.
        
        Args:
            k: NAL evidence horizon parameter.
        """
        self.k = k
        self.match_counts: Dict[str, int] = defaultdict(int)
        self.non_match_counts: Dict[str, int] = defaultdict(int)
        self.total_matches = 0
        self.total_non_matches = 0
    
    def add_observation(self, judgments: Set[str], is_match: bool):
        """
        Record an observation of judgments with label.
        
        Args:
            judgments: Set of judgment strings for a pair.
            is_match: Whether the pair is a true match.
        """
        if is_match:
            self.total_matches += 1
            for j in judgments:
                self.match_counts[j] += 1
        else:
            self.total_non_matches += 1
            for j in judgments:
                self.non_match_counts[j] += 1
    
    def get_judgment_weight(self, judgment: str) -> float:
        """
        Get log-likelihood ratio weight for a judgment.
        
        Positive weight = evidence for match.
        Negative weight = evidence against match.
        
        Args:
            judgment: The judgment string.
            
        Returns:
            Log-likelihood ratio (clamped).
        """
        m_count = self.match_counts.get(judgment, 0)
        nm_count = self.non_match_counts.get(judgment, 0)
        
        if self.total_matches == 0 or self.total_non_matches == 0:
            return 0.0
        
        p_j_given_match = (m_count + 1) / (self.total_matches + 2)
        p_j_given_nonmatch = (nm_count + 1) / (self.total_non_matches + 2)
        
        if p_j_given_nonmatch == 0:
            return 5.0
        
        log_ratio = np.log(p_j_given_match / p_j_given_nonmatch)
        return np.clip(log_ratio, -5.0, 5.0)
    
    def compute_score(self, judgments: Set[str]) -> float:
        """
        Compute match score for a set of judgments.
        
        Uses sum of log-likelihood ratios, then sigmoid.
        
        Args:
            judgments: Set of judgment strings.
            
        Returns:
            Score in [0, 1].
        """
        if not judgments:
            return 0.5
        
        log_odds = 0.0
        for j in judgments:
            log_odds += self.get_judgment_weight(j)
        
        base_rate = self.total_matches / (self.total_matches + self.total_non_matches + 1e-10)
        prior_log_odds = np.log(base_rate / (1 - base_rate + 1e-10))
        log_odds += prior_log_odds
        
        score = 1.0 / (1.0 + np.exp(-log_odds))
        return score


class NarsModel(BaseModel):
    """
    NARS model for entity resolution.
    
    Uses domain-specific preprocessing to generate judgments,
    then learns log-likelihood weights for each judgment type
    from training data.
    """
    
    def __init__(
        self,
        k: float = 1.0,
        preprocess: str = "iceid",
        exclude_judgments: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize NARS model.
        
        Args:
            k: NAL confidence parameter.
            preprocess: Preprocessing mode ("iceid", "generic", or "deepmatcher[:name]").
            exclude_judgments: List of judgment prefixes to exclude (for ablations).
        """
        super().__init__("nars", **kwargs)
        self.k = k
        self.preprocess_mode = preprocess
        self.dataset_name = None
        if ":" in preprocess:
            mode, _, name = preprocess.partition(":")
            self.preprocess_mode = mode
            self.dataset_name = name.strip() or None
        self.exclude_judgments = exclude_judgments or []
        self.stats = JudgmentStatistics(k=k)
        self.threshold = 0.5

    def _should_include(self, prefix: str) -> bool:
        for excl in self.exclude_judgments:
            if prefix.startswith(excl) or excl in prefix:
                return False
        return True

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        tokens = set()
        for tok in text.split():
            if len(tok) >= 2 or any(ch.isdigit() for ch in tok):
                tokens.add(tok)
        return tokens

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _parse_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None
        text = text.replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _parse_year(value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value)
        match = re.search(r"(1[7-9]\d{2}|20\d{2})", text)
        if not match:
            return None
        try:
            return int(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _parse_time_seconds(value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if ":" not in text:
            num = NarsModel._parse_number(text)
            return int(num) if num is not None else None
        parts = text.split(":")
        try:
            nums = [int(float(p)) for p in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            return nums[0] * 60 + nums[1]
        if len(nums) == 3:
            return nums[0] * 3600 + nums[1] * 60 + nums[2]
        return None

    @staticmethod
    def _digits_only(value: Any) -> str:
        if value is None:
            return ""
        return "".join(ch for ch in str(value) if ch.isdigit())

    def _add_text_judgments(self, judgments: Set[str], field: str, v1: Any, v2: Any):
        if not self._should_include(field):
            return
        n1 = self._normalize_text(v1)
        n2 = self._normalize_text(v2)
        if not n1 or not n2:
            return
        if n1 == n2:
            judgments.add(f"{field}_exact")
        tokens1 = self._tokenize(n1)
        tokens2 = self._tokenize(n2)
        jacc = self._jaccard(tokens1, tokens2)
        seq = self._string_similarity(n1, n2)
        sim = max(jacc, seq)
        if sim >= 0.9:
            judgments.add(f"{field}_sim_high")
        elif sim >= 0.75:
            judgments.add(f"{field}_sim_med")
        elif sim >= 0.5:
            judgments.add(f"{field}_sim_low")
        else:
            judgments.add(f"{field}_sim_vlow")

    def _add_numeric_judgments(
        self,
        judgments: Set[str],
        field: str,
        v1: Any,
        v2: Any,
        close_abs: Optional[float] = None,
        close_rel: Optional[float] = None,
        far_rel: Optional[float] = None,
        exact_abs: float = 0.01,
    ):
        if not self._should_include(field):
            return
        n1 = self._parse_number(v1)
        n2 = self._parse_number(v2)
        if n1 is None or n2 is None:
            return
        diff = abs(n1 - n2)
        rel = diff / max(abs(n1), abs(n2), 1.0)
        if diff <= exact_abs:
            judgments.add(f"{field}_exact")
        if (close_abs is not None and diff <= close_abs) or (close_rel is not None and rel <= close_rel):
            judgments.add(f"{field}_close")
        elif far_rel is not None and rel >= far_rel:
            judgments.add(f"{field}_far")

    def _add_year_judgments(self, judgments: Set[str], field: str, v1: Any, v2: Any, close_years: int = 1):
        if not self._should_include(field):
            return
        y1 = self._parse_year(v1)
        y2 = self._parse_year(v2)
        if y1 is None or y2 is None:
            return
        diff = abs(y1 - y2)
        if diff == 0:
            judgments.add(f"{field}_same")
        elif diff <= close_years:
            judgments.add(f"{field}_close")
        else:
            judgments.add(f"{field}_far")

    def _add_time_judgments(self, judgments: Set[str], field: str, v1: Any, v2: Any, close_seconds: int = 5):
        if not self._should_include(field):
            return
        t1 = self._parse_time_seconds(v1)
        t2 = self._parse_time_seconds(v2)
        if t1 is None or t2 is None:
            return
        diff = abs(t1 - t2)
        if diff == 0:
            judgments.add(f"{field}_same")
        elif diff <= close_seconds:
            judgments.add(f"{field}_close")
        else:
            judgments.add(f"{field}_far")

    def _add_phone_judgments(self, judgments: Set[str], v1: Any, v2: Any):
        if not self._should_include("phone"):
            return
        d1 = self._digits_only(v1)
        d2 = self._digits_only(v2)
        if len(d1) < 7 or len(d2) < 7:
            return
        if d1 == d2:
            judgments.add("phone_exact")
        elif d1[-7:] == d2[-7:]:
            judgments.add("phone_last7")
        else:
            judgments.add("phone_mismatch")

    def _add_address_number_judgments(self, judgments: Set[str], v1: Any, v2: Any):
        if not self._should_include("addr_num"):
            return
        n1 = self._parse_number(v1)
        n2 = self._parse_number(v2)
        if n1 is None or n2 is None:
            return
        if int(n1) == int(n2):
            judgments.add("addr_num_match")
        else:
            judgments.add("addr_num_diff")
    
    def _preprocess_iceid(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> Set[str]:
        """Generate judgments for ICE-ID records."""
        judgments = set()
        
        try:
            h1 = int(float(rec1.get("heimild", 0) or 0))
        except (ValueError, TypeError):
            h1 = 0
        try:
            h2 = int(float(rec2.get("heimild", 0) or 0))
        except (ValueError, TypeError):
            h2 = 0
        if h1 and h2 and self._should_include("heimild"):
            diff = abs(h1 - h2)
            if diff == 0:
                judgments.add("same_census")
            elif diff <= 10:
                judgments.add("adjacent_census")
            else:
                judgments.add("distant_census")
        
        for field in ["nafn_norm", "first_name", "patronym", "surname"]:
            if not self._should_include(field):
                continue
            v1 = str(rec1.get(field, "")).strip().lower()
            v2 = str(rec2.get(field, "")).strip().lower()
            if v1 and v2:
                if v1 == v2:
                    judgments.add(f"same_{field}")
                else:
                    judgments.add(f"different_{field}")
        
        try:
            by1 = int(float(rec1.get("birthyear", -1) or -1))
        except (ValueError, TypeError):
            by1 = -1
        try:
            by2 = int(float(rec2.get("birthyear", -1) or -1))
        except (ValueError, TypeError):
            by2 = -1
        if by1 > 0 and by2 > 0 and self._should_include("birthyear"):
            diff = abs(by1 - by2)
            if diff == 0:
                judgments.add("same_birthyear")
            elif diff <= 2:
                judgments.add("birthyear_very_close")
            elif diff <= 5:
                judgments.add("birthyear_close")
            else:
                judgments.add("different_birthyear")
        
        s1 = str(rec1.get("sex", "")).strip().lower()
        s2 = str(rec2.get("sex", "")).strip().lower()
        if s1 and s2 and self._should_include("sex"):
            if s1 == s2:
                judgments.add("same_sex")
            else:
                judgments.add("different_sex")
        
        for field in ["farm", "parish", "district", "county"]:
            if not self._should_include(field):
                continue
            v1 = str(rec1.get(field, "")).strip().lower()
            v2 = str(rec2.get(field, "")).strip().lower()
            if v1 and v2:
                if v1 == v2:
                    judgments.add(f"same_{field}")
                else:
                    judgments.add(f"different_{field}")
        
        return judgments

    def _preprocess_deepmatcher(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> Set[str]:
        """Generate judgments for DeepMatcher-style two-table datasets."""
        judgments = set()
        dataset = (self.dataset_name or "").lower().replace("-", "_")

        if dataset == "abt_buy":
            self._add_text_judgments(judgments, "name", rec1.get("name"), rec2.get("name"))
            self._add_text_judgments(judgments, "description", rec1.get("description"), rec2.get("description"))
            self._add_numeric_judgments(judgments, "price", rec1.get("price"), rec2.get("price"), close_abs=1.0, close_rel=0.05, far_rel=0.2)
        elif dataset == "amazon_google":
            self._add_text_judgments(judgments, "title", rec1.get("title"), rec2.get("title"))
            self._add_text_judgments(judgments, "manufacturer", rec1.get("manufacturer"), rec2.get("manufacturer"))
            self._add_numeric_judgments(judgments, "price", rec1.get("price"), rec2.get("price"), close_abs=1.0, close_rel=0.05, far_rel=0.2)
        elif dataset in {"dblp_acm", "dblp_scholar"}:
            self._add_text_judgments(judgments, "title", rec1.get("title"), rec2.get("title"))
            self._add_text_judgments(judgments, "authors", rec1.get("authors"), rec2.get("authors"))
            self._add_text_judgments(judgments, "venue", rec1.get("venue"), rec2.get("venue"))
            self._add_year_judgments(judgments, "year", rec1.get("year"), rec2.get("year"), close_years=1)
        elif dataset == "itunes_amazon":
            self._add_text_judgments(judgments, "song", rec1.get("Song_Name"), rec2.get("Song_Name"))
            self._add_text_judgments(judgments, "artist", rec1.get("Artist_Name"), rec2.get("Artist_Name"))
            self._add_text_judgments(judgments, "album", rec1.get("Album_Name"), rec2.get("Album_Name"))
            self._add_text_judgments(judgments, "genre", rec1.get("Genre"), rec2.get("Genre"))
            self._add_numeric_judgments(judgments, "price", rec1.get("Price"), rec2.get("Price"), close_abs=0.5, close_rel=0.05, far_rel=0.2)
            self._add_time_judgments(judgments, "time", rec1.get("Time"), rec2.get("Time"), close_seconds=5)
            self._add_year_judgments(judgments, "released", rec1.get("Released"), rec2.get("Released"), close_years=1)
            self._add_year_judgments(judgments, "copyright", rec1.get("CopyRight"), rec2.get("CopyRight"), close_years=1)
        elif dataset == "walmart_amazon":
            self._add_text_judgments(judgments, "title", rec1.get("title"), rec2.get("title"))
            self._add_text_judgments(judgments, "category", rec1.get("category"), rec2.get("category"))
            self._add_text_judgments(judgments, "brand", rec1.get("brand"), rec2.get("brand"))
            self._add_text_judgments(judgments, "modelno", rec1.get("modelno"), rec2.get("modelno"))
            self._add_numeric_judgments(judgments, "price", rec1.get("price"), rec2.get("price"), close_abs=1.0, close_rel=0.05, far_rel=0.2)
        elif dataset == "beer":
            self._add_text_judgments(judgments, "beer_name", rec1.get("Beer_Name"), rec2.get("Beer_Name"))
            self._add_text_judgments(judgments, "brew_factory", rec1.get("Brew_Factory_Name"), rec2.get("Brew_Factory_Name"))
            self._add_text_judgments(judgments, "style", rec1.get("Style"), rec2.get("Style"))
            self._add_numeric_judgments(judgments, "abv", rec1.get("ABV"), rec2.get("ABV"), close_abs=0.3, close_rel=0.05, far_rel=0.2)
        elif dataset == "fodors_zagats":
            self._add_text_judgments(judgments, "name", rec1.get("name"), rec2.get("name"))
            self._add_text_judgments(judgments, "addr", rec1.get("addr"), rec2.get("addr"))
            self._add_text_judgments(judgments, "city", rec1.get("city"), rec2.get("city"))
            self._add_text_judgments(judgments, "type", rec1.get("type"), rec2.get("type"))
            self._add_text_judgments(judgments, "class", rec1.get("class"), rec2.get("class"))
            self._add_phone_judgments(judgments, rec1.get("phone"), rec2.get("phone"))
            self._add_address_number_judgments(judgments, rec1.get("addr"), rec2.get("addr"))
        else:
            return self._preprocess_generic(rec1, rec2)

        return judgments
    
    def _preprocess_generic(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> Set[str]:
        """Generate judgments for generic records using token overlap."""
        judgments = set()
        
        tokens1 = set()
        tokens2 = set()
        
        for k, v in rec1.items():
            if k != "id":
                tokens1.update(str(v).lower().split())
        
        for k, v in rec2.items():
            if k != "id":
                tokens2.update(str(v).lower().split())
        
        intersection = tokens1 & tokens2
        
        if tokens1 or tokens2:
            overlap = len(intersection) / max(len(tokens1), len(tokens2), 1)
            if overlap > 0.7:
                judgments.add("very_high_overlap")
            elif overlap > 0.5:
                judgments.add("high_overlap")
            elif overlap > 0.3:
                judgments.add("medium_overlap")
            elif overlap > 0.1:
                judgments.add("low_overlap")
            else:
                judgments.add("very_low_overlap")
        
        for token in list(intersection)[:5]:
            if len(token) >= 4:
                judgments.add(f"shared_{token[:12]}")
        
        return judgments
    
    def _preprocess(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> Set[str]:
        """
        Generate judgments based on preprocessing mode.
        
        Strips label-leaking fields such as `person` from records before
        generating judgments.
        """
        if "person" in rec1:
            rec1 = {k: v for k, v in rec1.items() if k != "person"}
        if "person" in rec2:
            rec2 = {k: v for k, v in rec2.items() if k != "person"}
        if self.preprocess_mode == "iceid":
            return self._preprocess_iceid(rec1, rec2)
        if self.preprocess_mode == "deepmatcher":
            return self._preprocess_deepmatcher(rec1, rec2)
        return self._preprocess_generic(rec1, rec2)
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """
        Train NARS by learning judgment statistics from labeled pairs.
        
        Args:
            dataset: Dataset split with records.
            train_pairs: DataFrame with id1, id2, label columns.
            val_pairs: Optional validation pairs (unused in this implementation).
        """
        id1_col = "id1" if "id1" in train_pairs.columns else "ltable_id"
        id2_col = "id2" if "id2" in train_pairs.columns else "rtable_id"
        
        self.stats = JudgmentStatistics(k=self.k)
        
        for _, row in train_pairs.iterrows():
            id1 = int(row[id1_col])
            id2 = int(row[id2_col])
            label = int(row["label"])
            
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            judgments = self._preprocess(rec1, rec2)
            self.stats.add_observation(judgments, is_match=(label == 1))
        
        self._calibrate_threshold(dataset, train_pairs, id1_col, id2_col)
        self._is_fitted = True
    
    def _calibrate_threshold(
        self,
        dataset: DatasetSplit,
        pairs: pd.DataFrame,
        id1_col: str,
        id2_col: str,
    ):
        """
        Calibrate threshold using F1 optimization on training scores.
        
        Args:
            dataset: Dataset split with records.
            pairs: DataFrame with labeled pairs.
            id1_col: Name of first ID column.
            id2_col: Name of second ID column.
        """
        scores = []
        labels = []
        
        for _, row in pairs.iterrows():
            id1 = int(row[id1_col])
            id2 = int(row[id2_col])
            label = int(row["label"])
            
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            judgments = self._preprocess(rec1, rec2)
            score = self.stats.compute_score(judgments)
            scores.append(score)
            labels.append(label)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.linspace(0.1, 0.9, 81):
            preds = (scores >= threshold).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """
        Score pairs using learned judgment statistics.
        
        Args:
            dataset: Dataset split with records.
            pairs: List of (id1, id2) pairs to score.
            
        Returns:
            Array of scores in [0, 1].
        """
        scores = []
        
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            judgments = self._preprocess(rec1, rec2)
            score = self.stats.compute_score(judgments)
            scores.append(score)
        
        return np.array(scores)
    
    def fit_pair(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any],
        label: int
    ):
        """
        Add a single training pair observation.
        
        Args:
            rec1: First record as dict.
            rec2: Second record as dict.
            label: 1 for match, 0 for non-match.
        """
        judgments = self._preprocess(rec1, rec2)
        self.stats.add_observation(judgments, is_match=(label == 1))
        self._is_fitted = True
    
    def score_pair(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> float:
        """
        Score a single pair.
        
        Args:
            rec1: First record as dict.
            rec2: Second record as dict.
            
        Returns:
            Score in [0, 1].
        """
        judgments = self._preprocess(rec1, rec2)
        return self.stats.compute_score(judgments)
    
    def get_judgment_weights(self) -> Dict[str, float]:
        """
        Get learned weights for each judgment type.
        
        Returns:
            Dictionary mapping judgment strings to log-likelihood weights.
        """
        all_judgments = set(self.stats.match_counts.keys()) | set(self.stats.non_match_counts.keys())
        return {j: self.stats.get_judgment_weight(j) for j in sorted(all_judgments)}


get_registry("models").register("nars_inspired", NarsModel)
