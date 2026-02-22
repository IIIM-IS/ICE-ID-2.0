"""
Isotonic regression calibrator.

Non-parametric calibration method that learns a monotonic mapping.
"""

import numpy as np
from .base import Calibrator


class IsotonicCalibrator(Calibrator):
    """
    Isotonic regression calibrator.
    
    Learns a non-parametric monotonic function to map scores
    to calibrated probabilities.
    """
    
    def __init__(self):
        """Initialize isotonic calibrator."""
        self._x_max = None
        self._y_hat = None
        self.threshold = 0.5
        self._is_fitted = False
    
    def fit(self, val_scores: np.ndarray, val_labels: np.ndarray):
        """
        Fit isotonic regression on validation data.
        
        Args:
            val_scores: Validation scores
            val_labels: Validation binary labels
        """
        if len(val_scores) < 2:
            self._is_fitted = False
            return

        x = np.asarray(val_scores, dtype=float).reshape(-1)
        y = np.asarray(val_labels, dtype=float).reshape(-1)

        order = np.argsort(x, kind="mergesort")
        x_sorted = x[order]
        y_sorted = y[order]

        sums = []
        counts = []
        x_right = []

        for xi, yi in zip(x_sorted, y_sorted):
            sums.append(float(yi))
            counts.append(1)
            x_right.append(float(xi))

            while len(sums) >= 2:
                avg_prev = sums[-2] / counts[-2]
                avg_last = sums[-1] / counts[-1]
                if avg_prev <= avg_last:
                    break
                sums[-2] += sums[-1]
                counts[-2] += counts[-1]
                x_right[-2] = x_right[-1]
                sums.pop()
                counts.pop()
                x_right.pop()

        y_hat = np.array([s / c for s, c in zip(sums, counts)], dtype=float)
        y_hat = np.clip(y_hat, 0.0, 1.0)

        self._x_max = np.array(x_right, dtype=float)
        self._y_hat = y_hat
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
        idx = np.searchsorted(self._x_max, x, side="right")
        idx = np.clip(idx, 0, len(self._y_hat) - 1)
        return self._y_hat[idx]

