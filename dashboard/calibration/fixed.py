"""Fixed threshold calibrator."""

import numpy as np
from .base import Calibrator


class FixedThresholdCalibrator(Calibrator):
    """
    Simple fixed threshold calibrator.
    
    Uses a pre-specified threshold value (default 0.5).
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize fixed threshold calibrator.
        
        Args:
            threshold: Fixed threshold value
        """
        self.threshold = threshold
    
    def fit(self, val_scores: np.ndarray, val_labels: np.ndarray):
        """
        No-op for fixed threshold.
        
        Args:
            val_scores: Validation scores (ignored)
            val_labels: Validation labels (ignored)
        """
        pass
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict using fixed threshold.
        
        Args:
            scores: Prediction scores
            
        Returns:
            Binary predictions
        """
        return (scores >= self.threshold).astype(int)
    
    def get_threshold(self) -> float:
        """Get the threshold value."""
        return self.threshold

