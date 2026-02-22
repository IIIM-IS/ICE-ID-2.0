"""Platt scaling (logistic calibration)."""

import numpy as np

from .base import BaseCalibrator
from ..core.registry import get_registry


class PlattCalibrator(BaseCalibrator):
    """
    Platt scaling using logistic regression.
    
    Fits a sigmoid function to map scores to probabilities.
    """
    
    def __init__(self, **kwargs):
        super().__init__("platt", **kwargs)
        self.a = 0.0
        self.b = 0.0
        self.threshold = 0.5
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit logistic regression on scores."""
        scores = np.asarray(scores, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=float).reshape(-1)

        if len(np.unique(labels)) < 2 or len(scores) == 0:
            self._is_fitted = True
            return

        y_mean = float(np.clip(labels.mean(), 1e-6, 1 - 1e-6))
        self.a = 0.0
        self.b = float(np.log(y_mean / (1 - y_mean)))

        reg = 1e-3
        for _ in range(50):
            z = self.a * scores + self.b
            z = np.clip(z, -50, 50)
            p = 1.0 / (1.0 + np.exp(-z))

            w = p * (1 - p)
            da = float(np.sum((p - labels) * scores) + reg * self.a)
            db = float(np.sum(p - labels))

            daa = float(np.sum(w * scores * scores) + reg)
            dbb = float(np.sum(w) + reg)
            dab = float(np.sum(w * scores))

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
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply logistic calibration."""
        scores = np.asarray(scores, dtype=float).reshape(-1)
        z = self.a * scores + self.b
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))


get_registry("calibrators").register("platt", PlattCalibrator)

