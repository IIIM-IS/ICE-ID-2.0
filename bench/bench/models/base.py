"""
Base model interface for entity resolution.

All model adapters must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ..core.types import DatasetSplit, Pair, PredictionResult


class BaseModel(ABC):
    """
    Abstract base class for ER model adapters.
    
    Provides a consistent interface for training, scoring, and prediction.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize model.
        
        Args:
            name: Model identifier.
            **kwargs: Model-specific parameters.
        """
        self.name = name
        self.params = kwargs
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """
        Train the model on labeled pairs.
        
        Args:
            dataset: Dataset split containing records.
            train_pairs: DataFrame with columns [id1, id2, label] or 
                        [ltable_id, rtable_id, label].
            val_pairs: Optional validation pairs.
        """
        raise NotImplementedError
    
    @abstractmethod
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """
        Score candidate pairs.
        
        Args:
            dataset: Dataset split containing records.
            pairs: List of (id1, id2) pairs to score.
            
        Returns:
            Array of scores in [0, 1].
        """
        raise NotImplementedError
    
    def predict(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
        threshold: float = 0.5,
    ) -> PredictionResult:
        """
        Predict matches with threshold.
        
        Args:
            dataset: Dataset split.
            pairs: Pairs to classify.
            threshold: Decision threshold.
            
        Returns:
            PredictionResult with scores and predictions.
        """
        scores = self.score(dataset, pairs)
        predictions = (scores >= threshold).astype(int)
        
        return PredictionResult(
            pairs=pairs,
            scores=scores,
            predictions=predictions,
            threshold=threshold,
        )
    
    @property
    def is_fitted(self) -> bool:
        """Returns True if model has been trained."""
        return self._is_fitted
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {"name": self.name, **self.params}
    
    def save(self, path: str):
        """Save model to disk."""
        raise NotImplementedError(f"{self.name} does not support save()")
    
    def load(self, path: str):
        """Load model from disk."""
        raise NotImplementedError(f"{self.name} does not support load()")


class TextBasedModel(BaseModel):
    """
    Base class for models that use text representations.
    
    Provides helper methods for creating text features from records.
    """
    
    def __init__(self, name: str, text_fields: List[str] = None, **kwargs):
        """
        Initialize text-based model.
        
        Args:
            name: Model identifier.
            text_fields: Fields to use for text representation.
        """
        super().__init__(name, **kwargs)
        self.text_fields = text_fields or []
    
    def _get_text(self, record: Dict[str, Any]) -> str:
        """Create text representation of a record."""
        if self.text_fields:
            parts = [str(record.get(f, "")) for f in self.text_fields]
        else:
            parts = [str(v) for k, v in record.items() if k != "id"]
        return " ".join(parts)
    
    def _get_pair_text(
        self,
        dataset: DatasetSplit,
        id1: int,
        id2: int,
        sep: str = " [SEP] "
    ) -> str:
        """Create concatenated text for a pair."""
        rec1 = dataset.get_record_by_id(id1) or {}
        rec2 = dataset.get_record_by_id(id2) or {}
        return self._get_text(rec1) + sep + self._get_text(rec2)
    
    def _prepare_pair_texts(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> List[str]:
        """Prepare text features for all pairs."""
        return [self._get_pair_text(dataset, id1, id2) for id1, id2 in pairs]

