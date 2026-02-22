# ICE-ID Benchmarking Framework

This directory contains the core entity resolution benchmarking framework.

---

## Directory Structure

```
bench/
├── bench/                      # Python package
│   ├── core/                   # Core types and utilities
│   │   ├── types.py            # DatasetSplit, Pair, Record, etc.
│   │   ├── registry.py         # Plugin registry
│   │   └── random.py           # Seed management
│   │
│   ├── data/                   # Dataset providers
│   │   ├── base.py             # BaseDataset abstract class
│   │   ├── iceid.py            # ICE-ID loader
│   │   ├── deepmatcher.py      # DeepMatcher datasets
│   │   └── ...
│   │
│   ├── models/                 # Model adapters
│   │   ├── base.py             # BaseModel (fit, score, predict)
│   │   ├── nars.py             # NARS symbolic model
│   │   ├── ditto_adapter.py    # Ditto deep learning
│   │   └── ...
│   │
│   ├── blocking/               # Candidate generation
│   │   ├── token_blocking.py   # Token-based blocking
│   │   └── ...
│   │
│   ├── metrics/                # Evaluation metrics
│   │   ├── pairwise.py         # P, R, F1, AUC
│   │   ├── ranking.py          # P@k, R@k
│   │   └── clustering.py       # ARI, B³
│   │
│   ├── calibration/            # Score calibration
│   │   ├── platt.py            # Platt scaling
│   │   └── isotonic.py         # Isotonic regression
│   │
│   └── clustering/             # Entity clustering
│       ├── connected_components.py
│       └── hac.py
│
├── scripts/                    # Executable scripts
│   ├── run_experiments.py      # Main experiment runner
│   ├── run_nars_full_eval.py   # NARS evaluation
│   ├── generate_paper_artifacts.py  # Paper figures/tables
│   └── prepare_data.py         # Data download/prep
│
├── benchmark_results/          # Experiment outputs (CSV/JSON)
├── paper_artifacts/            # Paper-ready artifacts
├── deepmatcher_data/           # Classic ER datasets
└── external/                   # External model repos
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run NARS Evaluation

```bash
python scripts/run_nars_full_eval.py
```

### Generate Paper Artifacts

```bash
python scripts/generate_paper_artifacts.py all
```

---

## Usage Examples

### Load ICE-ID Dataset

```python
from bench.data.iceid import IceIdDataset

dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

print(f"Records: {len(split.records)}")
print(f"Unique persons: {len(split.cluster_labels)}")
print(f"Train pairs: {len(split.train_pairs)}")
```

### Train NARS Model

```python
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel

dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

model = NarsModel(preprocess='iceid')
model.fit(split, split.train_pairs, split.val_pairs)

# Score pairs
test_pairs = [(row['id1'], row['id2']) for _, row in split.test_pairs.iterrows()]
scores = model.score(split, test_pairs)
```

### Compute Metrics

```python
from bench.metrics.pairwise import compute_pairwise_metrics
from bench.metrics.ranking import compute_ranking_metrics

labels = split.test_pairs['label'].values

# Pairwise metrics
pairwise = compute_pairwise_metrics(labels, y_scores=scores, threshold=0.5)
print(f"F1: {pairwise['f1']:.3f}")
print(f"AUC: {pairwise['auc']:.3f}")

# Ranking metrics
n_pos = sum(labels)
ranking = compute_ranking_metrics(labels, scores, k_values=[n_pos])
print(f"P@k: {ranking[f'p_at_{n_pos}']:.3f}")
```

---

## Available Components

### Datasets

| Name | Class | Description |
|------|-------|-------------|
| ICE-ID | `IceIdDataset` | Icelandic census (984K records) |
| Abt-Buy | `DeepMatcherDataset` | Product matching |
| Amazon-Google | `DeepMatcherDataset` | Product matching |
| DBLP-ACM | `DeepMatcherDataset` | Citation matching |
| DBLP-Scholar | `DeepMatcherDataset` | Citation matching |

### Models

| Name | Class | Type |
|------|-------|------|
| NARS | `NarsModel` | Symbolic reasoning |
| Fellegi-Sunter | `FellegiSunterModel` | Probabilistic |
| Rules | `RulesModel` | Deterministic |
| Ditto | `DittoModel` | Deep learning |
| ZeroER | `ZeroERModel` | Unsupervised |

### Metrics

| Function | Description |
|----------|-------------|
| `compute_pairwise_metrics()` | P, R, F1, Acc, AUC |
| `compute_ranking_metrics()` | P@k, R@k |
| `compute_clustering_metrics()` | ARI, B³ |

---

## Output Directories

### `benchmark_results/`

Experiment outputs in CSV format:

```
benchmark_results/
├── nars_full_eval.csv          # NARS on all datasets
├── nars_ablations.csv          # Ablation study
├── ditto_results.csv           # Ditto results
└── ...
```

### `paper_artifacts/`

Paper-ready data:

```
paper_artifacts/
├── plot_data/                  # JSON for figures
│   ├── fig1_temporal_coverage.json
│   └── ...
├── table_data/                 # CSV for tables
│   ├── table1_dataset_synopsis.csv
│   └── ...
└── nars_full_eval.json         # Model results
```

---

## Configuration

Example YAML config (`config/examples/iceid_nars.yaml`):

```yaml
dataset:
  name: iceid
  data_dir: ../raw_data
  train_before_year: 1870

model:
  name: nars
  preprocess: iceid
  k: 1.0

blocking:
  name: token
  fields: [nafn_norm, birthyear]
  max_block_size: 200

metrics:
  - f1
  - auc
  - ari
```

Run with:

```bash
python -m bench.runner.run_one --config config/examples/iceid_nars.yaml
```

---

## External Models

External model repositories are in `external/`:

```bash
# Setup external models
cd external
./setup_external.sh
pip install -r requirements_external.txt
```

| Model | Directory | Source |
|-------|-----------|--------|
| Ditto | `external/ditto/` | github.com/megagonlabs/ditto |
| ZeroER | `external/zeroer/` | github.com/chu-data-lab/zeroer |
| AnyMatch | `external/anymatch/` | github.com/megagonlabs/anymatch |
| MatchGPT | `external/MatchGPT/` | github.com/wbsg-uni-mannheim/MatchGPT |
| OpenNARS | `external/OpenNARS-for-Applications/` | github.com/opennars/OpenNARS-for-Applications |

---

## Development

### Adding a New Model

1. Create `bench/models/my_model.py`:

```python
from .base import BaseModel
from ..core.registry import get_registry

class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__("my_model", **kwargs)
    
    def fit(self, dataset, train_pairs, val_pairs=None):
        # Training logic
        self._is_fitted = True
    
    def score(self, dataset, pairs):
        # Return numpy array of scores
        return scores

get_registry("models").register("my_model", MyModel)
```

2. Add to `bench/models/__init__.py`

### Adding a New Dataset

1. Create `bench/data/my_dataset.py`:

```python
from .base import BaseDataset
from ..core.types import DatasetSplit
from ..core.registry import get_registry

class MyDataset(BaseDataset):
    def load(self) -> DatasetSplit:
        # Load data and create splits
        return DatasetSplit(
            name="my_dataset",
            records=records_df,
            train_pairs=train_df,
            val_pairs=val_df,
            test_pairs=test_df,
        )

get_registry("datasets").register("my_dataset", MyDataset)
```

2. Add to `bench/data/__init__.py`

---

## Testing

```bash
# Quick smoke test
python -c "
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel

ds = IceIdDataset(data_dir='../raw_data')
split = ds.load()
print(f'Loaded {len(split.records)} records')

model = NarsModel(preprocess='iceid')
print('Model OK')
"
```

---

## See Also

- [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) — Full codebase map
- [../docs/WORKFLOWS.md](../docs/WORKFLOWS.md) — Step-by-step guides
- [../docs/BENCHMARK_GUIDE.md](../docs/BENCHMARK_GUIDE.md) — Detailed benchmarking guide
