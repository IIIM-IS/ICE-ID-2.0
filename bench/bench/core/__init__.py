"""Core types, registry, and utilities."""

from .types import Record, Pair, ScoredPair, DatasetSplit, ClusterLabels
from .registry import Registry, get_registry
from .random import set_seed, get_rng

