# ICE-ID Documentation

This folder contains comprehensive documentation for the ICE-ID entity resolution benchmark.

---

## Documentation Index

| Document | Description | When to Read |
|----------|-------------|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete map of all files and folders | Understanding the codebase structure |
| [WORKFLOWS.md](WORKFLOWS.md) | Step-by-step guides for common tasks | Doing specific operations |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | How to run benchmarks and evaluate models | Running experiments |

---

## Quick Links

### Getting Started
- [Main README](../README.md) — Project overview
- [QUICK_START.md](../QUICK_START.md) — 5-minute setup guide

### Running Experiments
- [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) — Full benchmarking guide
- [bench/README.md](../bench/README.md) — Benchmark framework docs

### Understanding the Code
- [ARCHITECTURE.md](ARCHITECTURE.md) — File/folder map
- [WORKFLOWS.md](WORKFLOWS.md) — How-to guides

---

## Documentation Structure

```
docs/
├── README.md           # This file (index)
├── ARCHITECTURE.md     # Complete file/folder map
│   ├── Top-level structure
│   ├── bench/ package breakdown
│   ├── Data flow diagrams
│   └── Design patterns
├── WORKFLOWS.md        # Step-by-step guides
│   ├── Running NARS evaluation
│   ├── Evaluating external models
│   ├── Generating paper artifacts
│   ├── Adding new datasets
│   ├── Adding new models
│   ├── Running ablations
│   ├── Compiling papers
│   └── Using the dashboard
└── BENCHMARK_GUIDE.md  # Benchmarking reference
    ├── Datasets available
    ├── Models available
    ├── Metrics explained
    ├── Running benchmarks
    ├── Blocking strategies
    ├── Calibration methods
    ├── Clustering algorithms
    └── Troubleshooting
```

---

## Key Concepts

### DatasetSplit

The central data container:

```python
@dataclass
class DatasetSplit:
    name: str                           # Dataset identifier
    records: Optional[pd.DataFrame]     # For deduplication
    left_table: Optional[pd.DataFrame]  # For two-table ER
    right_table: Optional[pd.DataFrame]
    train_pairs: Optional[pd.DataFrame] # [(id1, id2, label), ...]
    val_pairs: Optional[pd.DataFrame]
    test_pairs: Optional[pd.DataFrame]
    cluster_labels: Optional[Dict]      # {record_id: cluster_id}
```

### BaseModel Interface

All models implement:

```python
class BaseModel:
    def fit(self, dataset, train_pairs, val_pairs=None):
        """Train on labeled pairs."""
        pass
    
    def score(self, dataset, pairs) -> np.ndarray:
        """Return match probabilities for pairs."""
        pass
```

### Plugin Registry

Components register themselves:

```python
from bench.core.registry import get_registry

# Register a model
get_registry("models").register("my_model", MyModel)

# Retrieve a model
model_cls = get_registry("models").get("my_model")
```

---

## Common Commands

```bash
# Full evaluation pipeline
cd bench
python scripts/run_nars_full_eval.py
python scripts/generate_paper_artifacts.py all
cd ../papers && pdflatex main_nars_paper.tex

# Quick smoke test
python -c "from bench.data.iceid import IceIdDataset; print('OK')"

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Need Help?

1. Check [WORKFLOWS.md](WORKFLOWS.md) for step-by-step guides
2. Check [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for troubleshooting
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the codebase
