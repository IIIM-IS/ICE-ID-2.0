"""
Fellegi-Sunter probabilistic record linkage.

Classic probabilistic matching based on agreement patterns across fields.
Computes m and u probabilities and uses likelihood ratio for decisions.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict
import recordlinkage as rl


class FellegiSunterMatcher:
    """
    Fellegi-Sunter probabilistic record linkage matcher.
    
    Learns m-probabilities (agreement given match) and u-probabilities
    (agreement given non-match) from training data, then computes
    log-likelihood ratios for classification.
    """
    
    def __init__(
        self,
        comparison_fields: List[str] = None,
        agreement_threshold: float = 0.85,
    ):
        """
        Initialize Fellegi-Sunter matcher.
        
        Args:
            comparison_fields: Fields to compare (default: auto-detect)
            agreement_threshold: Similarity threshold for "agreement"
        """
        self.comparison_fields = comparison_fields
        self.agreement_threshold = agreement_threshold
        
        # Learned probabilities
        self.m_probs: Dict[str, float] = {}
        self.u_probs: Dict[str, float] = {}
        self.threshold: float = 0.0
    
    def fit(
        self,
        left_table: pd.DataFrame,
        right_table: pd.DataFrame,
        pairs: pd.DataFrame,
    ):
        """
        Learn m and u probabilities from labeled pairs.
        
        Args:
            left_table: Left entity table
            right_table: Right entity table (or same as left for dedup)
            pairs: Labeled pairs with columns [ltable_id, rtable_id, label]
        """
        # Auto-detect comparison fields if not provided
        if self.comparison_fields is None:
            self.comparison_fields = self._detect_fields(left_table, right_table)
        
        # Set up record linkage comparison
        compare_cl = rl.Compare()
        
        for field in self.comparison_fields:
            if field in left_table.columns and field in right_table.columns:
                # Use Jaro-Winkler for strings
                compare_cl.string(
                    field, field,
                    method='jarowinkler',
                    threshold=self.agreement_threshold,
                    label=field
                )
        
        # Compute comparison vectors for training pairs
        left_idx = left_table.set_index("id")
        right_idx = right_table.set_index("id")
        
        # Create multiindex for pairs
        pair_index = pd.MultiIndex.from_arrays([
            pairs["ltable_id"].values,
            pairs["rtable_id"].values
        ])
        
        features = compare_cl.compute(pair_index, left_idx, right_idx)
        
        # Split into matches and non-matches
        is_match = pairs["label"] == 1
        match_features = features[is_match.values]
        nonmatch_features = features[~is_match.values]
        
        # Estimate m and u probabilities
        for field in features.columns:
            # m-probability: P(agreement | match)
            if len(match_features) > 0:
                self.m_probs[field] = match_features[field].mean()
            else:
                self.m_probs[field] = 0.9  # default high
            
            # u-probability: P(agreement | non-match)
            if len(nonmatch_features) > 0:
                self.u_probs[field] = nonmatch_features[field].mean()
            else:
                self.u_probs[field] = 0.1  # default low
            
            # Avoid zero probabilities
            self.m_probs[field] = max(0.01, min(0.99, self.m_probs[field]))
            self.u_probs[field] = max(0.01, min(0.99, self.u_probs[field]))
        
        # Calibrate threshold on training data
        self._calibrate_threshold(features, pairs["label"].values)
    
    def _detect_fields(
        self,
        left_table: pd.DataFrame,
        right_table: pd.DataFrame
    ) -> List[str]:
        """Auto-detect comparable string fields."""
        common_fields = set(left_table.columns) & set(right_table.columns)
        
        # Exclude ID and metadata fields
        exclude = {"id", "text", "label", "cluster_id"}
        
        string_fields = []
        for field in common_fields:
            if field not in exclude:
                # Check if mostly string data
                if left_table[field].dtype == object:
                    string_fields.append(field)
        
        return string_fields
    
    def _calibrate_threshold(self, features: pd.DataFrame, labels: np.ndarray):
        """Calibrate decision threshold on training data."""
        scores = self.compute_scores(features)
        
        # Find threshold that maximizes F1
        best_f1 = 0
        best_threshold = 0
        
        for threshold in np.linspace(scores.min(), scores.max(), 100):
            preds = (scores >= threshold).astype(int)
            
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            
            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        self.threshold = best_threshold
    
    def compute_scores(self, features: pd.DataFrame) -> np.ndarray:
        """
        Compute log-likelihood ratio scores for feature vectors.
        
        Args:
            features: Comparison feature matrix
            
        Returns:
            Array of log-likelihood ratio scores
        """
        scores = np.zeros(len(features))
        
        for field in features.columns:
            if field in self.m_probs and field in self.u_probs:
                agreements = features[field].values
                
                m = self.m_probs[field]
                u = self.u_probs[field]
                
                # Log-likelihood ratio
                # log(P(agree|match) / P(agree|non-match)) if agree
                # log(P(disagree|match) / P(disagree|non-match)) if disagree
                for i, agree in enumerate(agreements):
                    if agree:
                        if u > 0:
                            scores[i] += np.log(m / u)
                    else:
                        if 1 - u > 0:
                            scores[i] += np.log((1 - m) / (1 - u))
        
        return scores
    
    def predict(
        self,
        left_table: pd.DataFrame,
        right_table: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict matches for candidate pairs.
        
        Args:
            left_table: Left entity table
            right_table: Right entity table
            candidate_pairs: Candidate pairs with columns [ltable_id, rtable_id]
            
        Returns:
            Binary predictions
        """
        scores = self.score_pairs(left_table, right_table, candidate_pairs)
        return (scores >= self.threshold).astype(int)
    
    def score_pairs(
        self,
        left_table: pd.DataFrame,
        right_table: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
    ) -> np.ndarray:
        """
        Score candidate pairs.
        
        Args:
            left_table: Left entity table
            right_table: Right entity table
            candidate_pairs: Candidate pairs with columns [ltable_id, rtable_id]
            
        Returns:
            Array of scores
        """
        # Set up comparison
        compare_cl = rl.Compare()
        
        for field in self.comparison_fields:
            if field in left_table.columns and field in right_table.columns:
                compare_cl.string(
                    field, field,
                    method='jarowinkler',
                    threshold=self.agreement_threshold,
                    label=field
                )
        
        # Compute features
        left_idx = left_table.set_index("id")
        right_idx = right_table.set_index("id")
        
        pair_index = pd.MultiIndex.from_arrays([
            candidate_pairs["ltable_id"].values,
            candidate_pairs["rtable_id"].values
        ])
        
        features = compare_cl.compute(pair_index, left_idx, right_idx)
        
        # Compute scores
        scores = self.compute_scores(features)
        
        return scores

