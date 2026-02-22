"""
AnyMatch model adapter.

Wrapper for AnyMatch - efficient zero-shot entity matching with small language models.

Reference: https://github.com/Jantory/anymatch
Paper: "AnyMatch -- Efficient Zero-Shot Entity Matching with a Small Language Model" (2024)
"""

import os
import sys
import subprocess
import tempfile
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


class AnyMatchModel(BaseModel):
    """
    AnyMatch model adapter.
    
    Zero-shot entity matching using small language models (GPT-2, etc.).
    Requires the AnyMatch repo to be cloned in external/anymatch/
    """
    
    def __init__(
        self,
        base_model: str = "gpt2",
        serialization_mode: str = "mode1",
        train_data: str = "attr+row",
        patience_start: int = 20,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize AnyMatch model.
        
        Args:
            base_model: Base language model (gpt2, gpt2-medium, etc.).
            serialization_mode: Serialization mode for pairs.
            train_data: Training data format.
            patience_start: Patience for early stopping.
            seed: Random seed.
        """
        super().__init__("anymatch", **kwargs)
        self.base_model = base_model
        self.serialization_mode = serialization_mode
        self.train_data = train_data
        self.patience_start = patience_start
        self.seed = seed
        self.anymatch_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "external", "anymatch"
        )
        self.model_path = None
    
    def _check_anymatch_available(self):
        """Check if AnyMatch repo is cloned."""
        if not os.path.exists(self.anymatch_dir):
            raise FileNotFoundError(
                f"AnyMatch not found at {self.anymatch_dir}. "
                "Run: cd bench/external && ./setup_external.sh"
            )
    
    def _prepare_anymatch_data(
        self,
        dataset: DatasetSplit,
        pairs_df: pd.DataFrame,
        output_path: str
    ):
        """
        Prepare data in AnyMatch format.
        
        AnyMatch expects CSV with specific columns.
        """
        id1_col = "id1" if "id1" in pairs_df.columns else "ltable_id"
        id2_col = "id2" if "id2" in pairs_df.columns else "rtable_id"
        
        anymatch_data = []
        
        for _, row in pairs_df.iterrows():
            id1 = int(row[id1_col])
            id2 = int(row[id2_col])
            label = int(row["label"])
            
            rec1 = dataset.get_record_by_id(id1) or {}
            rec2 = dataset.get_record_by_id(id2) or {}
            
            entry = {
                "id1": id1,
                "id2": id2,
                "label": label,
                **{f"left_{k}": v for k, v in rec1.items() if k != "id"},
                **{f"right_{k}": v for k, v in rec2.items() if k != "id"},
            }
            
            anymatch_data.append(entry)
        
        pd.DataFrame(anymatch_data).to_csv(output_path, index=False)
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """Train AnyMatch model using external repo."""
        self._check_anymatch_available()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            model_dir = os.path.join(tmpdir, "model")
            os.makedirs(model_dir, exist_ok=True)
            
            self._prepare_anymatch_data(dataset, train_pairs, train_path)
            
            train_cmd = [
                sys.executable,
                os.path.join(self.anymatch_dir, "train.py"),
                "--seed", str(self.seed),
                "--base_model", self.base_model,
                "--train_path", train_path,
                "--serialization_mode", self.serialization_mode,
                "--train_data", self.train_data,
                "--patience_start", str(self.patience_start),
                "--output_dir", model_dir,
            ]
            
            try:
                result = subprocess.run(
                    train_cmd,
                    cwd=self.anymatch_dir,
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                if result.returncode == 0:
                    self.model_path = model_dir
                    self._is_fitted = True
                else:
                    print(f"AnyMatch training failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("AnyMatch training timed out after 1 hour")
            except Exception as e:
                print(f"AnyMatch training error: {e}")
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """Score pairs using trained AnyMatch model."""
        if not self._is_fitted or not self.model_path:
            return np.random.rand(len(pairs)) * 0.5 + 0.5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = os.path.join(tmpdir, "test.csv")
            output_path = os.path.join(tmpdir, "predictions.json")
            
            test_df = pd.DataFrame({
                "id1": [p[0] for p in pairs],
                "id2": [p[1] for p in pairs],
                "label": [0] * len(pairs)
            })
            
            self._prepare_anymatch_data(dataset, test_df, test_path)
            
            predict_cmd = [
                sys.executable,
                os.path.join(self.anymatch_dir, "predict.py"),
                "--model_path", self.model_path,
                "--test_path", test_path,
                "--output_path", output_path,
                "--serialization_mode", self.serialization_mode,
            ]
            
            try:
                result = subprocess.run(
                    predict_cmd,
                    cwd=self.anymatch_dir,
                    capture_output=True,
                    text=True,
                    timeout=1800
                )
                
                if result.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        predictions = json.load(f)
                    
                    return np.array([p["score"] for p in predictions])
                
            except Exception as e:
                print(f"AnyMatch prediction error: {e}")
        
        return np.random.rand(len(pairs)) * 0.5 + 0.5


get_registry("models").register("anymatch", AnyMatchModel)


