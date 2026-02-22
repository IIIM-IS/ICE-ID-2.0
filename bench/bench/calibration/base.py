"""Base calibrator interface."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseCalibrator(ABC):
    """
    Abstract base class for score calibration.
    
    Calibrators transform raw model scores into calibrated
    probabilities or determine optimal thresholds.
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        self._is_fitted = False
        self.threshold = 0.5
    
    @abstractmethod
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit calibrator on validation data.
        
        Args:
            scores: Model confidence scores.
            labels: True binary labels.
        """
        raise NotImplementedError
    
    @abstractmethod
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores to calibrated probabilities.
        
        Args:
            scores: Raw model scores.
            
        Returns:
            Calibrated probabilities.
        """
        raise NotImplementedError
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            scores: Raw model scores.
            
        Returns:
            Binary predictions.
        """
        calibrated = self.calibrate(scores)
        return (calibrated >= self.threshold).astype(int)
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

