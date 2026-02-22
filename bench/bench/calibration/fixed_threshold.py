"""Fixed threshold calibration (NARS median-based)."""

import numpy as np

from .base import BaseCalibrator
from ..core.registry import get_registry


class FixedThresholdCalibrator(BaseCalibrator):
    """
    Fixed threshold calibration.
    
    Uses median of positive and negative scores to set threshold.
    """
    
    def __init__(self, default_threshold: float = 0.5, **kwargs):
        super().__init__("fixed_threshold", **kwargs)
        self.default_threshold = default_threshold
        self.threshold = default_threshold
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit threshold using median approach."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            self.threshold = (np.median(pos_scores) + np.median(neg_scores)) / 2
        elif len(pos_scores) > 0:
            self.threshold = np.median(pos_scores) * 0.9
        elif len(neg_scores) > 0:
            self.threshold = np.median(neg_scores) * 1.1
        else:
            self.threshold = self.default_threshold
        
        self._is_fitted = True
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Return original scores (no transformation)."""
        return scores


get_registry("calibrators").register("fixed_threshold", FixedThresholdCalibrator)

