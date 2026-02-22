"""Blocking strategies for candidate generation."""

from .base import BlockingStrategy
from .trivial import TrivialBlocking
from .token_blocking import TokenBlocking
from .phonetic_blocking import PhoneticBlocking
from .geo_hierarchy import GeoHierarchyBlocking

__all__ = [
    "BlockingStrategy",
    "TrivialBlocking",
    "TokenBlocking",
    "PhoneticBlocking",
    "GeoHierarchyBlocking",
]

