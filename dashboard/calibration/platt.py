"""
Platt scaling calibrator.

Uses logistic regression to calibrate scores on validation set.
"""

import numpy as np
from .base import Calibrator


class PlattCalibrator(Calibrator):
    """
    Platt scaling calibrator.
    
    Fits a logistic regression model to map scores to probabilities,
    then uses 0.5 as decision boundary on calibrated scores.
    """
    
    def __init__(self):
        """Initialize Platt calibrator."""
        self.a = 0.0
        self.b = 0.0
        self.threshold = 0.5
        self._is_fitted = False
    
    def fit(self, val_scores: np.ndarray, val_labels: np.ndarray):
        """
        Fit logistic regression on validation data.
        
        Args:
            val_scores: Validation scores
            val_labels: Validation binary labels
        """
        if len(val_scores) == 0:
            self._is_fitted = False
            return

        x = np.asarray(val_scores, dtype=float).reshape(-1)
        y = np.asarray(val_labels, dtype=float).reshape(-1)

        if np.unique(y).size < 2:
            self._is_fitted = False
            return

        y_mean = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.a = 0.0
        self.b = float(np.log(y_mean / (1 - y_mean)))

        reg = 1e-3
        for _ in range(50):
            z = self.a * x + self.b
            z = np.clip(z, -50, 50)
            p = 1.0 / (1.0 + np.exp(-z))

            w = p * (1 - p)
            da = float(np.sum((p - y) * x) + reg * self.a)
            db = float(np.sum(p - y))

            daa = float(np.sum(w * x * x) + reg)
            dbb = float(np.sum(w) + reg)
            dab = float(np.sum(w * x))

            det = daa * dbb - dab * dab
            if det <= 1e-12:
                break

            step_a = (dbb * da - dab * db) / det
            step_b = (-dab * da + daa * db) / det

            self.a -= 0.5 * step_a
            self.b -= 0.5 * step_b

            if abs(step_a) + abs(step_b) < 1e-8:
                break

        self._is_fitted = True
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Predict using calibrated scores.
        
        Args:
            scores: Prediction scores
            
        Returns:
            Binary predictions
        """
        if not self._is_fitted:
            # Fallback to fixed threshold
            return (scores >= 0.5).astype(int)

        probs = self.calibrate_scores(scores)
        return (probs >= self.threshold).astype(int)
    
    def get_threshold(self) -> float:
        """Get the threshold value."""
        return self.threshold
    
    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Get calibrated probability scores.
        
        Args:
            scores: Raw prediction scores
            
        Returns:
            Calibrated probabilities
        """
        if not self._is_fitted:
            return scores

        x = np.asarray(scores, dtype=float).reshape(-1)
        z = self.a * x + self.b
        z = np.clip(z, -50, 50)
        p = 1.0 / (1.0 + np.exp(-z))
        return p

