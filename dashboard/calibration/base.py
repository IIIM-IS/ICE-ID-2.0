"""
Base calibrator interface.

All calibrators must implement this interface for threshold selection.
"""

from abc import ABC, abstractmethod
import numpy as np


class Calibrator(ABC):
    """Abstract base class for score calibrators."""
    
    @abstractmethod
    def fit(self, val_scores: np.ndarray, val_labels: np.ndarray):
        """
        Fit calibrator on validation data.
        
        Args:
            val_scores: Validation scores
            val_labels: Validation labels (binary)
        """
        pass
    
    @abstractmethod
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict binary labels for scores.
        
        Args:
            scores: Prediction scores
            
        Returns:
            Binary predictions (0 or 1)
        """
        pass
    
    @abstractmethod
    def get_threshold(self) -> float:
        """
        Get the calibrated threshold.
        
        Returns:
            Threshold value
        """
        pass

