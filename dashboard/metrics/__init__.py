"""Metrics module for entity resolution evaluation."""

from .pairwise import compute_pairwise_metrics
from .clustering import compute_clustering_metrics, validate_clustering_inputs
from .ranking import compute_ranking_metrics

__all__ = [
    "compute_pairwise_metrics",
    "compute_clustering_metrics",
    "validate_clustering_inputs",
    "compute_ranking_metrics",
]

