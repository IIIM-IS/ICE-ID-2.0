"""
ER-Bench: Extensible Entity Resolution Benchmark Framework

Provides standardized interfaces for datasets, models, blocking,
calibration, clustering, and evaluation metrics.
"""

__version__ = "0.1.0"

from .core.types import Record, Pair, ScoredPair, DatasetSplit, ClusterLabels
from .core.registry import Registry, get_registry

from . import data
from . import models
from . import blocking
from . import calibration
from . import clustering
from . import metrics

