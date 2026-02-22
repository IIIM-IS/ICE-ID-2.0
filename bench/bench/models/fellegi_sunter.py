"""
Fellegi-Sunter probabilistic record linkage model.

Classical model that estimates m-probabilities (agreement given match)
and u-probabilities (agreement given non-match) to compute likelihood ratios.

Reference: Fellegi & Sunter (1969). A Theory for Record Linkage.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import pandas as pd

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


class FellegiSunterModel(BaseModel):
    """
    Fellegi-Sunter probabilistic record linkage.
    
    Learns m/u probabilities for field agreement patterns
    and scores pairs using log-likelihood ratios.
    """
    
    def __init__(
        self,
        comparison_fields: List[str] = None,
        smoothing: float = 0.1,
        **kwargs
    ):
        """
        Initialize Fellegi-Sunter model.
        
        Args:
            comparison_fields: Fields to compare between records.
            smoothing: Laplace smoothing for probability estimates.
        """
        super().__init__("fellegi_sunter", **kwargs)
        self.comparison_fields = comparison_fields or []
        self.smoothing = smoothing
        self.m_probs: Dict[str, Dict[str, float]] = {}
        self.u_probs: Dict[str, Dict[str, float]] = {}
        self.threshold = 0.0
    
    def _get_agreement_pattern(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> Dict[str, str]:
        """Compute agreement pattern for a pair."""
        pattern = {}
        
        fields = self.comparison_fields or list(
            set(rec1.keys()) & set(rec2.keys()) - {"id"}
        )
        
        for field in fields:
            v1 = str(rec1.get(field, "")).strip().lower()
            v2 = str(rec2.get(field, "")).strip().lower()
            
            if not v1 or not v2:
                pattern[field] = "missing"
            elif v1 == v2:
                pattern[field] = "exact_match"
            else:
                pattern[field] = "mismatch"
        
        return pattern
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """Learn m and u probabilities from training data."""
        id1_col = "id1" if "id1" in train_pairs.columns else "ltable_id"
        id2_col = "id2" if "id2" in train_pairs.columns else "rtable_id"
        
        m_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        u_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        total_matches = 0
        total_non_matches = 0
        
        for _, row in train_pairs.iterrows():
            id1 = int(row[id1_col])
            id2 = int(row[id2_col])
            label = int(row["label"])
            
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            pattern = self._get_agreement_pattern(rec1, rec2)
            
            for field, agreement in pattern.items():
                if label == 1:
                    m_counts[field][agreement] += 1
                else:
                    u_counts[field][agreement] += 1
            
            if label == 1:
                total_matches += 1
            else:
                total_non_matches += 1
        
        if total_matches == 0 or total_non_matches == 0:
            self._is_fitted = True
            return
        
        all_fields = set(m_counts.keys()) | set(u_counts.keys())
        
        for field in all_fields:
            self.m_probs[field] = {}
            self.u_probs[field] = {}
            
            all_agreements = set(m_counts[field].keys()) | set(u_counts[field].keys())
            n_agreements = len(all_agreements)
            
            for agreement in all_agreements:
                m_num = m_counts[field][agreement] + self.smoothing
                m_den = total_matches + self.smoothing * n_agreements
                self.m_probs[field][agreement] = m_num / m_den
                
                u_num = u_counts[field][agreement] + self.smoothing
                u_den = total_non_matches + self.smoothing * n_agreements
                self.u_probs[field][agreement] = u_num / u_den
        
        self._calibrate_threshold(dataset, train_pairs, id1_col, id2_col)
        self._is_fitted = True
    
    def _calibrate_threshold(
        self,
        dataset: DatasetSplit,
        pairs: pd.DataFrame,
        id1_col: str,
        id2_col: str,
    ):
        """Find threshold that maximizes F1 on training data."""
        scores = []
        labels = []
        
        for _, row in pairs.iterrows():
            id1 = int(row[id1_col])
            id2 = int(row[id2_col])
            label = int(row["label"])
            
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            score = self._score_pair(rec1, rec2)
            scores.append(score)
            labels.append(label)
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            self.threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
        else:
            self.threshold = 0.0
    
    def _score_pair(
        self,
        rec1: Dict[str, Any],
        rec2: Dict[str, Any]
    ) -> float:
        """Compute log-likelihood ratio for a pair."""
        pattern = self._get_agreement_pattern(rec1, rec2)
        
        log_lr = 0.0
        
        for field, agreement in pattern.items():
            m = self.m_probs.get(field, {}).get(agreement, 0.5)
            u = self.u_probs.get(field, {}).get(agreement, 0.5)
            
            m = max(m, 1e-10)
            u = max(u, 1e-10)
            
            log_lr += math.log(m / u)
        
        return log_lr
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """Score pairs using log-likelihood ratio."""
        scores = []
        
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            scores.append(self._score_pair(rec1, rec2))
        
        scores = np.array(scores)
        scores = 1.0 / (1.0 + np.exp(-scores))
        
        return scores


get_registry("models").register("fellegi_sunter", FellegiSunterModel)

