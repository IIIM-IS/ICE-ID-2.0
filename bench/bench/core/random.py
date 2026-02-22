"""
Random state management for reproducibility.

All sampling operations should use get_rng() to ensure determinism.
"""

import random
import numpy as np
from typing import Optional

_global_seed: int = 42
_global_rng: Optional[np.random.Generator] = None


def set_seed(seed: int = 42):
    """Set the global random seed for reproducibility."""
    global _global_seed, _global_rng
    _global_seed = seed
    _global_rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_rng() -> np.random.Generator:
    """Get the global random number generator."""
    global _global_rng
    if _global_rng is None:
        set_seed(_global_seed)
    return _global_rng


def get_seed() -> int:
    """Get the current global seed."""
    return _global_seed


# Initialize with default seed
set_seed(42)

