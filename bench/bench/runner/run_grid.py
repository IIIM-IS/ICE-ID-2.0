"""Run grid of experiments."""

import os
import itertools
from typing import Dict, Any, List
import pandas as pd
import yaml

from .run_one import run_experiment
from ..core.types import EvaluationResult


def run_grid(
    grid_config: Dict[str, Any],
    output_dir: str = "results",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run a grid of experiments.
    
    Args:
        grid_config: Configuration with grid parameters.
        output_dir: Base output directory.
        verbose: Print progress.
        
    Returns:
        DataFrame with results for all experiments.
    """
    base_config = grid_config.get("base", {})
    grid_params = grid_config.get("grid", {})
    
    param_names = list(grid_params.keys())
    param_values = [grid_params[k] if isinstance(grid_params[k], list) else [grid_params[k]] 
                   for k in param_names]
    
    all_results = []
    
    for i, values in enumerate(itertools.product(*param_values)):
        config = _deep_copy(base_config)
        
        for name, value in zip(param_names, values):
            _set_nested(config, name, value)
        
        config["name"] = f"exp_{i:04d}"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}/{len(list(itertools.product(*param_values)))}")
            params_str = ", ".join(f"{n}={v}" for n, v in zip(param_names, values))
            print(f"  {params_str}")
        
        exp_output = os.path.join(output_dir, config["name"])
        
        try:
            result = run_experiment(config, exp_output, verbose=verbose)
            
            row = result.to_dict()
            for name, value in zip(param_names, values):
                row[f"param_{name}"] = value
            
            all_results.append(row)
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            all_results.append({
                "error": str(e),
                **{f"param_{name}": value for name, value in zip(param_names, values)}
            })
    
    results_df = pd.DataFrame(all_results)
    
    results_df.to_csv(os.path.join(output_dir, "grid_results.csv"), index=False)
    
    return results_df


def _deep_copy(d: Dict) -> Dict:
    """Deep copy a dictionary."""
    import copy
    return copy.deepcopy(d)


def _set_nested(d: Dict, key: str, value: Any):
    """Set a nested key in a dict (e.g., 'model.params.pool_size')."""
    keys = key.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

