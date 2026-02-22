"""
ZeroER model adapter.

Wrapper for ZeroER - unsupervised entity resolution without labeled examples.

Reference: https://github.com/chu-data-lab/zeroer
Paper: "ZeroER: Entity Resolution using Zero Labeled Examples" (SIGMOD 2020)
"""

import os
import sys
import subprocess
import tempfile
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry


class ZeroERModel(BaseModel):
    """
    ZeroER model adapter.
    
    Unsupervised entity resolution that doesn't require labeled training data.
    Requires the ZeroER repo to be cloned in external/zeroer/
    """
    
    def __init__(
        self,
        run_transitivity: bool = True,
        **kwargs
    ):
        """
        Initialize ZeroER model.
        
        Args:
            run_transitivity: Whether to run transitivity closure for clustering.
        """
        super().__init__("zeroer", **kwargs)
        self.run_transitivity = run_transitivity
        self.zeroer_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "external", "zeroer"
        )
        self.results = {}
    
    def _check_zeroer_available(self):
        """Check if ZeroER repo is cloned."""
        if not os.path.exists(self.zeroer_dir):
            raise FileNotFoundError(
                f"ZeroER not found at {self.zeroer_dir}. "
                "Run: cd bench/external && ./setup_external.sh"
            )
    
    def _prepare_zeroer_dataset(
        self,
        dataset: DatasetSplit,
        dataset_dir: str
    ):
        """
        Prepare dataset in ZeroER format.
        
        ZeroER expects:
        - metadata.txt: column names
        - tableA.csv or single table.csv
        - tableB.csv (for two-table matching)
        """
        os.makedirs(dataset_dir, exist_ok=True)
        
        if dataset.left_table is not None and dataset.right_table is not None:
            left = dataset.left_table.copy()
            right = dataset.right_table.copy()
            
            left.to_csv(
                os.path.join(dataset_dir, "tableA.csv"),
                index=False
            )
            right.to_csv(
                os.path.join(dataset_dir, "tableB.csv"),
                index=False
            )
            
            columns = [c for c in left.columns if c != "id"]
            
        elif dataset.records is not None:
            records = dataset.records.copy()
            records.to_csv(
                os.path.join(dataset_dir, "table.csv"),
                index=False
            )
            
            columns = [c for c in records.columns if c != "id"]
        else:
            raise ValueError("Dataset must have either records or left/right tables")
        
        with open(os.path.join(dataset_dir, "metadata.txt"), 'w') as f:
            f.write("\\n".join(columns))
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: Optional[pd.DataFrame] = None,
    ):
        """
        ZeroER is unsupervised, so fit is a no-op.
        
        We mark as fitted to indicate the model is ready.
        """
        self._check_zeroer_available()
        self._is_fitted = True
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: List[Pair],
    ) -> np.ndarray:
        """
        Score pairs using ZeroER.
        
        Since ZeroER is unsupervised and doesn't provide continuous scores,
        we run it once and cache the results.
        """
        self._check_zeroer_available()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            self._prepare_zeroer_dataset(dataset, dataset_dir)
            
            dataset_name = os.path.basename(dataset_dir)
            
            zeroer_cmd = [
                sys.executable,
                os.path.join(self.zeroer_dir, "zeroer.py"),
                dataset_name
            ]
            
            if self.run_transitivity:
                zeroer_cmd.append("--run_transitivity")
            
            try:
                result = subprocess.run(
                    zeroer_cmd,
                    cwd=self.zeroer_dir,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    env={**os.environ, "DATASETS_DIR": tmpdir}
                )
                
                if result.returncode == 0:
                    output_file = os.path.join(
                        self.zeroer_dir,
                        "output",
                        f"{dataset_name}_output.csv"
                    )
                    
                    if os.path.exists(output_file):
                        results_df = pd.read_csv(output_file)
                        
                        for _, row in results_df.iterrows():
                            pair = (int(row["id1"]), int(row["id2"]))
                            self.results[pair] = float(row.get("similarity", 0.5))
                
            except subprocess.TimeoutExpired:
                print("ZeroER timed out after 30 minutes")
            except Exception as e:
                print(f"ZeroER error: {e}")
        
        scores = []
        for id1, id2 in pairs:
            pair_key = (id1, id2)
            reverse_key = (id2, id1)
            
            if pair_key in self.results:
                scores.append(self.results[pair_key])
            elif reverse_key in self.results:
                scores.append(self.results[reverse_key])
            else:
                scores.append(0.3)
        
        return np.array(scores)


get_registry("models").register("zeroer", ZeroERModel)


