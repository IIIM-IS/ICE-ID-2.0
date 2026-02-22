"""Configuration schema and validation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import yaml


@dataclass
class DatasetConfig:
    name: str
    data_dir: str = None
    train_before_year: int = None
    val_year_range: tuple = None
    test_year_range: tuple = None


@dataclass
class BlockingConfig:
    name: str = "trivial_allpairs"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairsConfig:
    mode: str = "from_clusters"
    train_ratio: float = 0.7
    cap: int = 50000
    negatives_per_positive: int = 2


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationConfig:
    name: str = "fixed_threshold"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusteringConfig:
    name: str = "connected_components"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    dataset: DatasetConfig = None
    blocking: BlockingConfig = None
    pairs: PairsConfig = None
    model: ModelConfig = None
    calibration: CalibrationConfig = None
    clustering: ClusteringConfig = None


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate configuration, return list of errors."""
    errors = []
    
    if "dataset" not in config:
        errors.append("Missing 'dataset' section")
    elif "name" not in config["dataset"]:
        errors.append("Missing 'dataset.name'")
    
    if "model" not in config:
        errors.append("Missing 'model' section")
    elif "name" not in config["model"]:
        errors.append("Missing 'model.name'")
    
    return errors

