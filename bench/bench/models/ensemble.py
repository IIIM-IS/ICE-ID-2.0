"""
ML Ensemble models for entity resolution.

Implements XGBoost, LightGBM, and RandomForest using TF-IDF features.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .base import BaseModel, TextBasedModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


class MLEnsembleModel(TextBasedModel):
    """
    Base ML ensemble model using TF-IDF features.
    
    Creates pairwise features by concatenating TF-IDF representations
    of both records in a pair.
    """
    
    def __init__(
        self,
        name: str,
        text_fields: List[str] = None,
        max_features: int = 1000,
        **kwargs
    ):
        super().__init__(name, text_fields, **kwargs)
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = None
    
    def _create_pair_features(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair]
    ) -> np.ndarray:
        """Create feature vectors for pairs using TF-IDF."""
        texts = []
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            text1 = self._get_text(rec1)
            text2 = self._get_text(rec2)
            
            texts.append(text1 + " " + text2)
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """Train the ML model on TF-IDF features."""
        id1_col = "id1" if "id1" in train_pairs.columns else "ltable_id"
        id2_col = "id2" if "id2" in train_pairs.columns else "rtable_id"
        
        pairs = list(zip(train_pairs[id1_col], train_pairs[id2_col]))
        labels = train_pairs["label"].values
        
        if len(np.unique(labels)) < 2:
            print(f"Warning: {self.name} requires at least 2 classes, got {len(np.unique(labels))}")
            self._is_fitted = False
            return
        
        all_texts = []
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(int(id1)) or {}
            rec2 = dataset.get_record_by_id(int(id2)) or {}
            
            text1 = self._get_text(rec1)
            text2 = self._get_text(rec2)
            
            all_texts.append(text1 + " " + text2)
        
        self.vectorizer.fit(all_texts)
        X = self.vectorizer.transform(all_texts).toarray()
        
        self._train_model(X, labels)
        self._is_fitted = True
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Override in subclasses to train specific model."""
        raise NotImplementedError
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """Score pairs using the trained model."""
        X = self._create_pair_features(dataset, pairs)
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]
        else:
            return self.model.predict(X).astype(float)


class XGBoostModel(MLEnsembleModel):
    """XGBoost model for entity resolution."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        if not HAS_XGB:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        super().__init__("xgboost", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost classifier."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X, y)


class LightGBMModel(MLEnsembleModel):
    """LightGBM model for entity resolution."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        if not HAS_LGB:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        super().__init__("lightgbm", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train LightGBM classifier."""
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X, y)


class RandomForestModel(MLEnsembleModel):
    """Random Forest model for entity resolution."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        **kwargs
    ):
        super().__init__("random_forest", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest classifier."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)


class GradientBoostingModel(MLEnsembleModel):
    """Gradient Boosting model for entity resolution."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        super().__init__("gradient_boosting", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train Gradient Boosting classifier."""
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        self.model.fit(X, y)


if HAS_XGB:
    get_registry("models").register("xgboost", XGBoostModel)
if HAS_LGB:
    get_registry("models").register("lightgbm", LightGBMModel)
get_registry("models").register("random_forest", RandomForestModel)
get_registry("models").register("gradient_boosting", GradientBoostingModel)


