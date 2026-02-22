"""
OpenNARS for Applications (ONA) model adapter.

Uses the real OpenNARS-for-Applications C implementation for entity resolution.
Communicates with ONA via subprocess stdin/stdout.

Reference: https://github.com/opennars/OpenNARS-for-Applications
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


@dataclass
class ONATruthValue:
    """Truth value parsed from ONA output."""
    frequency: float
    confidence: float
    
    @property
    def expectation(self) -> float:
        return self.confidence * (self.frequency - 0.5) + 0.5


class OpenNARSModel(BaseModel):
    """
    Entity resolution using real OpenNARS-for-Applications.
    
    Converts record pairs to Narsese statements, feeds them to ONA,
    and queries for similarity judgments.
    """
    
    def __init__(
        self,
        cycles_per_query: int = 50,
        volume: int = 0,
        preprocess: str = "iceid",
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize OpenNARS model.
        
        Args:
            cycles_per_query: Inference cycles to run per query.
            volume: ONA output volume (0=quiet, 100=verbose).
            preprocess: Preprocessing mode ("iceid" or "generic").
            timeout: Timeout in seconds for ONA subprocess.
        """
        super().__init__("opennars", **kwargs)
        self.cycles_per_query = cycles_per_query
        self.volume = volume
        self.preprocess_mode = preprocess
        self.timeout = timeout
        self.ona_path = os.path.join(
            os.path.dirname(__file__), "..", "..", 
            "external", "OpenNARS-for-Applications", "NAR"
        )
        self.learned_patterns: List[str] = []
        self.threshold = 0.5
    
    def _check_ona_available(self) -> bool:
        """Check if ONA executable exists."""
        if not os.path.exists(self.ona_path):
            raise FileNotFoundError(
                f"ONA not found at {self.ona_path}. "
                "Run: cd bench/external/OpenNARS-for-Applications && ./build.sh"
            )
        return True
    
    def _sanitize_term(self, s: str) -> str:
        """Sanitize string for Narsese (alphanumeric + underscore only)."""
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(s).strip().lower())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized[:30] if sanitized else "unknown"
    
    def _record_to_narsese(self, rec: Dict[str, Any], rec_id: str) -> List[str]:
        """Convert a record to Narsese statements."""
        statements = []
        for key, val in rec.items():
            if key == "id" or val is None or str(val).strip() == "":
                continue
            sanitized_key = self._sanitize_term(key)
            sanitized_val = self._sanitize_term(str(val))
            if sanitized_val:
                statements.append(f"<{{{rec_id}}} --> [{sanitized_key}_{sanitized_val}]>. :|:")
        return statements
    
    def _generate_judgments_iceid(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any],
        id1: str,
        id2: str
    ) -> List[str]:
        """Generate Narsese judgments for ICE-ID record pair comparison."""
        statements = []
        
        for field in ["nafn_norm", "first_name", "patronym", "surname"]:
            v1 = str(rec1.get(field, "")).strip().lower()
            v2 = str(rec2.get(field, "")).strip().lower()
            if v1 and v2:
                if v1 == v2:
                    statements.append(f"<({{{id1}}} * {{{id2}}}) --> [same_{field}]>. :|:")
                else:
                    statements.append(f"<({{{id1}}} * {{{id2}}}) --> [diff_{field}]>. :|:")
        
        try:
            by1 = int(float(rec1.get("birthyear", -1) or -1))
        except (ValueError, TypeError):
            by1 = -1
        try:
            by2 = int(float(rec2.get("birthyear", -1) or -1))
        except (ValueError, TypeError):
            by2 = -1
        if by1 > 0 and by2 > 0:
            if by1 == by2:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [same_birthyear]>. :|:")
            elif abs(by1 - by2) <= 5:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [close_birthyear]>. :|:")
            else:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [diff_birthyear]>. :|:")
        
        s1 = str(rec1.get("sex", "")).strip().lower()
        s2 = str(rec2.get("sex", "")).strip().lower()
        if s1 and s2:
            if s1 == s2:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [same_sex]>. :|:")
            else:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [diff_sex]>. :|:")
        
        for field in ["farm", "parish", "district", "county"]:
            v1 = str(rec1.get(field, "")).strip().lower()
            v2 = str(rec2.get(field, "")).strip().lower()
            if v1 and v2:
                if v1 == v2:
                    statements.append(f"<({{{id1}}} * {{{id2}}}) --> [same_{field}]>. :|:")
        
        return statements
    
    def _generate_judgments_generic(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any],
        id1: str,
        id2: str
    ) -> List[str]:
        """Generate Narsese judgments for generic record pair comparison."""
        statements = []
        
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
            if overlap > 0.5:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [high_overlap]>. :|:")
            elif overlap > 0.2:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [medium_overlap]>. :|:")
            else:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [low_overlap]>. :|:")
        
        for token in list(intersection)[:5]:
            sanitized = self._sanitize_term(token)
            if sanitized:
                statements.append(f"<({{{id1}}} * {{{id2}}}) --> [shared_{sanitized}]>. :|:")
        
        return statements
    
    def _generate_judgments(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any],
        id1: str,
        id2: str
    ) -> List[str]:
        """Generate Narsese judgments based on preprocessing mode."""
        if self.preprocess_mode == "iceid":
            return self._generate_judgments_iceid(rec1, rec2, id1, id2)
        return self._generate_judgments_generic(rec1, rec2, id1, id2)
    
    def _parse_truth_value(self, output: str) -> Optional[ONATruthValue]:
        """Parse truth value from ONA output."""
        pattern = r'Truth:\s*frequency=([\d.]+)\s*confidence=([\d.]+)'
        match = re.search(pattern, output)
        if match:
            return ONATruthValue(
                frequency=float(match.group(1)),
                confidence=float(match.group(2))
            )
        
        pattern2 = r'\{([\d.]+)\s+([\d.]+)\}'
        match2 = re.search(pattern2, output)
        if match2:
            return ONATruthValue(
                frequency=float(match2.group(1)),
                confidence=float(match2.group(2))
            )
        
        return None
    
    def _run_ona_batch(self, narsese_input: str) -> str:
        """Run ONA with batch input and return output."""
        self._check_ona_available()
        
        try:
            result = subprocess.run(
                [self.ona_path, "shell"],
                input=narsese_input,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return ""
        except Exception as e:
            print(f"ONA error: {e}")
            return ""
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """
        Train OpenNARS by learning patterns from labeled pairs.
        
        Feeds training examples as Narsese statements to ONA to learn
        the relationship between attribute comparisons and match labels.
        """
        self._check_ona_available()
        
        id1_col = "id1" if "id1" in train_pairs.columns else "ltable_id"
        id2_col = "id2" if "id2" in train_pairs.columns else "rtable_id"
        
        narsese_lines = [f"*volume={self.volume}"]
        
        narsese_lines.append("<[same_first_name] ==> match>. %0.9;0.9%")
        narsese_lines.append("<[same_surname] ==> match>. %0.8;0.9%")
        narsese_lines.append("<[same_birthyear] ==> match>. %0.8;0.9%")
        narsese_lines.append("<[close_birthyear] ==> match>. %0.6;0.8%")
        narsese_lines.append("<[same_sex] ==> match>. %0.6;0.7%")
        narsese_lines.append("<[same_parish] ==> match>. %0.7;0.8%")
        narsese_lines.append("<[same_county] ==> match>. %0.6;0.7%")
        narsese_lines.append("<[diff_birthyear] ==> match>. %0.2;0.9%")
        narsese_lines.append("<[diff_sex] ==> match>. %0.1;0.9%")
        narsese_lines.append("<[diff_first_name] ==> match>. %0.3;0.8%")
        narsese_lines.append("<[high_overlap] ==> match>. %0.9;0.9%")
        narsese_lines.append("<[medium_overlap] ==> match>. %0.5;0.8%")
        narsese_lines.append("<[low_overlap] ==> match>. %0.2;0.8%")
        
        for _, row in train_pairs.iterrows():
            raw_id1 = row[id1_col]
            raw_id2 = row[id2_col]
            try:
                id1 = int(float(str(raw_id1).replace("L_", "").replace("R_", "")))
                id2 = int(float(str(raw_id2).replace("L_", "").replace("R_", "")))
            except (ValueError, TypeError):
                continue
            
            label = int(row["label"])
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            id1_str = f"rec{id1}"
            id2_str = f"rec{id2}"
            
            judgments = self._generate_judgments(rec1, rec2, id1_str, id2_str)
            narsese_lines.extend(judgments)
            
            freq = "0.9" if label == 1 else "0.1"
            narsese_lines.append(f"<({{{id1_str}}} * {{{id2_str}}}) --> match>. %{freq};0.9%")
            
            self.learned_patterns.extend(judgments)
        
        narsese_lines.append(str(self.cycles_per_query * len(train_pairs) // 10))
        
        narsese_input = "\n".join(narsese_lines)
        self._run_ona_batch(narsese_input)
        
        self._calibrate_threshold(dataset, train_pairs, id1_col, id2_col)
        self._is_fitted = True
    
    def _calibrate_threshold(
        self,
        dataset: DatasetSplit,
        pairs: pd.DataFrame,
        id1_col: str,
        id2_col: str,
    ):
        """Calibrate threshold using training data."""
        pos_scores = []
        neg_scores = []
        
        sample_size = min(100, len(pairs))
        sample_pairs = pairs.sample(n=sample_size, random_state=42) if len(pairs) > sample_size else pairs
        
        scores = self._score_batch(dataset, sample_pairs, id1_col, id2_col)
        
        for i, (_, row) in enumerate(sample_pairs.iterrows()):
            label = int(row["label"])
            if label == 1:
                pos_scores.append(scores[i])
            else:
                neg_scores.append(scores[i])
        
        if pos_scores and neg_scores:
            self.threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
        elif pos_scores:
            self.threshold = np.median(pos_scores) * 0.9
        else:
            self.threshold = 0.5
    
    def _score_batch(
        self,
        dataset: DatasetSplit,
        pairs_df: pd.DataFrame,
        id1_col: str,
        id2_col: str
    ) -> List[float]:
        """Score a batch of pairs using ONA."""
        scores = []
        narsese_lines = [f"*volume={self.volume}"]
        
        narsese_lines.append("<[same_first_name] ==> match>. %0.9;0.9%")
        narsese_lines.append("<[same_surname] ==> match>. %0.8;0.9%")
        narsese_lines.append("<[same_birthyear] ==> match>. %0.8;0.9%")
        narsese_lines.append("<[close_birthyear] ==> match>. %0.6;0.8%")
        narsese_lines.append("<[same_sex] ==> match>. %0.6;0.7%")
        narsese_lines.append("<[same_parish] ==> match>. %0.7;0.8%")
        narsese_lines.append("<[diff_birthyear] ==> match>. %0.2;0.9%")
        narsese_lines.append("<[diff_sex] ==> match>. %0.1;0.9%")
        narsese_lines.append("<[high_overlap] ==> match>. %0.9;0.9%")
        narsese_lines.append("<[medium_overlap] ==> match>. %0.5;0.8%")
        narsese_lines.append("<[low_overlap] ==> match>. %0.2;0.8%")
        
        query_markers = []
        
        for _, row in pairs_df.iterrows():
            raw_id1 = row[id1_col]
            raw_id2 = row[id2_col]
            try:
                id1 = int(float(str(raw_id1).replace("L_", "").replace("R_", "")))
                id2 = int(float(str(raw_id2).replace("L_", "").replace("R_", "")))
            except (ValueError, TypeError):
                scores.append(0.5)
                continue
            
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            id1_str = f"rec{id1}"
            id2_str = f"rec{id2}"
            
            judgments = self._generate_judgments(rec1, rec2, id1_str, id2_str)
            narsese_lines.extend(judgments)
            
            narsese_lines.append(str(self.cycles_per_query))
            
            query = f"<({{{id1_str}}} * {{{id2_str}}}) --> match>?"
            narsese_lines.append(query)
            query_markers.append(query)
        
        narsese_input = "\n".join(narsese_lines)
        output = self._run_ona_batch(narsese_input)
        
        if not output:
            return [0.5] * len(pairs_df)
        
        output_lines = output.split("\n")
        query_idx = 0
        
        for line in output_lines:
            if "Answer:" in line or "-->match" in line.replace(" ", ""):
                tv = self._parse_truth_value(line)
                if tv:
                    scores.append(tv.expectation)
                    query_idx += 1
                    if query_idx >= len(pairs_df):
                        break
        
        while len(scores) < len(pairs_df):
            scores.append(0.5)
        
        return scores[:len(pairs_df)]
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """Score pairs using OpenNARS pattern matching."""
        if not pairs:
            return np.array([])
        
        pairs_df = pd.DataFrame({
            "id1": [p[0] for p in pairs],
            "id2": [p[1] for p in pairs]
        })
        
        scores = self._score_batch(dataset, pairs_df, "id1", "id2")
        return np.array(scores)


get_registry("models").register("opennars", OpenNARSModel)
get_registry("models").register("nars", OpenNARSModel)

