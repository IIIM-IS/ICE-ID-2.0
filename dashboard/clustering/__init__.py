"""Clustering algorithms for entity resolution."""

from .base import Clusterer
from .connected_components import ConnectedComponentsClusterer
from .hac import HACClusterer

__all__ = [
    "Clusterer",
    "ConnectedComponentsClusterer",
    "HACClusterer",
]

