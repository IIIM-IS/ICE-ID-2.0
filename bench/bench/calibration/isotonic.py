"""Isotonic regression calibration."""

import numpy as np

from .base import BaseCalibrator
from ..core.registry import get_registry


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic regression calibration.
    
    Non-parametric calibration that preserves monotonicity.
    """
    
    def __init__(self, **kwargs):
        super().__init__("isotonic", **kwargs)
        self._x_max = None
        self._y_hat = None
        self.threshold = 0.5
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression."""
        scores = np.asarray(scores, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=float).reshape(-1)

        if len(np.unique(labels)) < 2 or len(scores) < 2:
            self._is_fitted = True
            return

        order = np.argsort(scores, kind="mergesort")
        x_sorted = scores[order]
        y_sorted = labels[order]

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
    
    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if self._x_max is None or self._y_hat is None:
            return scores

        x = np.asarray(scores, dtype=float).reshape(-1)
        idx = np.searchsorted(self._x_max, x, side="right")
        idx = np.clip(idx, 0, len(self._y_hat) - 1)
        return self._y_hat[idx]


get_registry("calibrators").register("isotonic", IsotonicCalibrator)

