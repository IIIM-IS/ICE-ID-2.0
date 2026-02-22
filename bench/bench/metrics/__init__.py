"""Evaluation metrics for entity resolution."""

from .pairwise import compute_pairwise_metrics
from .clustering import compute_clustering_metrics, compute_b3_metrics
from .ranking import compute_ranking_metrics
from .sanity import run_sanity_checks

