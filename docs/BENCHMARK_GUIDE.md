# Benchmarking Guide

Complete guide to evaluating entity resolution models with the ICE-ID benchmarking framework.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Models](#-models)
- [Metrics](#-metrics)
- [Running Benchmarks](#-running-benchmarks)
- [Blocking Strategies](#-blocking-strategies)
- [Calibration Methods](#-calibration-methods)
- [Clustering](#-clustering)
- [Output Formats](#-output-formats)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

```bash
cd ICE-ID-2.0/bench

# Install dependencies
pip install -r requirements.txt

# Run NARS on all datasets
python scripts/run_nars_full_eval.py
```

---

## 📊 Datasets

### ICE-ID (Main Dataset)

| Property | Value |
|----------|-------|
| **Records** | 984,028 |
| **Unique Persons** | 226,864 |
| **Time Span** | 1703–1920 (16 waves) |
| **Task** | Longitudinal person linkage |

**Temporal Splits:**
- **Train**: pre-1870 (560,334 records)
- **Validation**: 1870–1890 (147,450 records)
- **Test**: 1891–1920 (276,244 records)

### DeepMatcher Datasets

| Dataset | Records | Positives | Domain |
|---------|---------|-----------|--------|
| **Abt-Buy** | 11,486 | 1,097 | Products |
| **Amazon-Google** | 13,748 | 1,167 | Products |
| **DBLP-ACM** | 14,834 | 2,220 | Citations |
| **DBLP-Scholar** | 34,446 | 5,347 | Citations |
| **Walmart-Amazon** | 12,288 | 962 | Products |
| **iTunes-Amazon** | 642 | 132 | Music |
| **Beer** | 536 | 68 | Beverages |
| **Fodors-Zagats** | 1,134 | 110 | Restaurants |

---

## 🤖 Models

### Internal Models

| Model | Type | Training Required | Description |
|-------|------|-------------------|-------------|
| `NarsModel` | Symbolic | ✅ Yes | Non-Axiomatic Reasoning |
| `FellegiSunterModel` | Probabilistic | ✅ Yes | Classic probabilistic linkage |
| `RulesModel` | Deterministic | ❌ No | Rule-based matching |
| `XGBoostModel` | ML | ✅ Yes | XGBoost gradient boosting |
| `LightGBMModel` | ML | ✅ Yes | LightGBM gradient boosting |
| `RandomForestModel` | ML | ✅ Yes | Random forest ensemble |
| `GradientBoostingModel` | ML | ✅ Yes | Scikit-learn gradient boosting |

### External Model Adapters

| Model | Type | Training Required | GPU Needed |
|-------|------|-------------------|------------|
| `DittoModel` | Deep Learning | ✅ Yes | ✅ Yes |
| `HierGATModel` | Graph Neural | ✅ Yes | ✅ Yes |
| `ZeroERModel` | Unsupervised | ❌ No | ❌ No |
| `AnyMatchModel` | Zero-shot | ❌ No | ✅ Yes |
| `MatchGPTModel` | LLM | ❌ No | ❌ No (API) |

---

## 📈 Metrics

### Pairwise Metrics

```
Precision (P) = TP / (TP + FP)
    How many predicted matches are correct

Recall (R) = TP / (TP + FN)
    How many true matches were found

F1 = 2 * P * R / (P + R)
    Harmonic mean of precision and recall

Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Overall correctness (less useful when imbalanced)

AUC = Area under ROC curve
    Ranking quality across all thresholds
```

### Ranking Metrics

```
P@k = (# positives in top k) / k
    Precision at top-k predictions

R@k = (# positives in top k) / (total positives)
    Recall at top-k predictions
```

When `k = # positives`, P@k = R@k (by design).

### Clustering Metrics

```
ARI (Adjusted Rand Index)
    Measures agreement between true and predicted clusters
    Range: [-1, 1], where 1 = perfect, 0 = random

B³ F1
    Cluster-level precision and recall
    Better for skewed cluster sizes
```

---

## 🔬 Running Benchmarks

### Method 1: Full Evaluation Script

```bash
python scripts/run_nars_full_eval.py
```

Evaluates NARS on ICE-ID + all DeepMatcher datasets.

### Method 2: Single Dataset Evaluation

```python
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel
from bench.metrics.pairwise import compute_pairwise_metrics

# Load dataset
dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

# Create train/val/test pairs
train_pairs = split.train_pairs
val_pairs = split.val_pairs
test_pairs = split.test_pairs

# Train model
model = NarsModel(preprocess='iceid')
model.fit(split, train_pairs, val_pairs)

# Score test pairs
test_pair_ids = [(row['id1'], row['id2']) for _, row in test_pairs.iterrows()]
scores = model.score(split, test_pair_ids)

# Compute metrics
labels = test_pairs['label'].values
metrics = compute_pairwise_metrics(labels, y_scores=scores, threshold=model.threshold)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"AUC: {metrics['auc']:.3f}")
```

### Method 3: Config-Driven Runs

Create a YAML config:

```yaml
# config.yaml
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

calibration:
  name: median_midpoint

metrics:
  - f1
  - auc
  - ari
```

Run:

```bash
python -m bench.runner.run_one --config config.yaml
```

---

## 🔄 Comparing Models

### Side-by-Side Evaluation

```python
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel
from bench.models.fellegi_sunter import FellegiSunterModel
from bench.metrics.pairwise import compute_pairwise_metrics

dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

models = {
    'NARS': NarsModel(preprocess='iceid'),
    'Fellegi-Sunter': FellegiSunterModel(),
}

results = []
for name, model in models.items():
    model.fit(split, split.train_pairs, split.val_pairs)
    
    test_pairs = [(row['id1'], row['id2']) for _, row in split.test_pairs.iterrows()]
    scores = model.score(split, test_pairs)
    labels = split.test_pairs['label'].values
    
    metrics = compute_pairwise_metrics(labels, y_scores=scores, threshold=model.threshold)
    results.append({
        'model': name,
        **metrics
    })

import pandas as pd
df = pd.DataFrame(results)
print(df[['model', 'precision', 'recall', 'f1', 'auc']])
```

---

## 🎯 Blocking Strategies

Blocking reduces the O(n²) comparison space.

### Available Blockers

```python
from bench.blocking.token_blocking import TokenBlocker
from bench.blocking.phonetic_blocking import PhoneticBlocker
from bench.blocking.geo_hierarchy import GeoHierarchyBlocker

# Token blocking on name
blocker = TokenBlocker(fields=['nafn_norm'], max_block_size=200)

# Phonetic blocking (Soundex)
blocker = PhoneticBlocker(fields=['first_name'], method='soundex')

# Geographic blocking
blocker = GeoHierarchyBlocker(levels=['parish', 'district', 'county'])
```

### Evaluate Blocking Recall

```python
from bench.blocking.token_blocking import TokenBlocker
from bench.data.iceid import IceIdDataset

dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

# Get ground truth pairs
true_pairs, _ = dataset.get_ground_truth()
true_pairs_set = set(tuple(sorted(p)) for p in true_pairs)

# Run blocking
blocker = TokenBlocker(fields=['nafn_norm'], max_block_size=200)
candidate_pairs = blocker.block(split)

# Compute recall
candidate_set = set(tuple(sorted(p)) for p in candidate_pairs.pairs)
found = len(true_pairs_set & candidate_set)
recall = found / len(true_pairs_set)

print(f"Blocking recall: {recall:.3f}")
print(f"Candidates per record: {len(candidate_pairs) / len(split.records):.1f}")
```

---

## 🎚️ Calibration Methods

Threshold calibration converts scores to binary predictions.

### Available Methods

```python
from bench.calibration.fixed_threshold import FixedThresholdCalibrator
from bench.calibration.platt import PlattCalibrator
from bench.calibration.isotonic import IsotonicCalibrator

# Fixed threshold
calibrator = FixedThresholdCalibrator(threshold=0.5)

# Platt scaling (sigmoid fit on validation)
calibrator = PlattCalibrator()
calibrator.fit(val_scores, val_labels)
calibrated_scores = calibrator.calibrate(test_scores)

# Isotonic regression
calibrator = IsotonicCalibrator()
calibrator.fit(val_scores, val_labels)
calibrated_scores = calibrator.calibrate(test_scores)
```

---

## 🔗 Clustering

Convert pairwise predictions to entity clusters.

### Available Clusterers

```python
from bench.clustering.connected_components import ConnectedComponentsClusterer
from bench.clustering.hac import HACClusterer

# Connected components on thresholded graph
clusterer = ConnectedComponentsClusterer(threshold=0.5)
clusters = clusterer.cluster(pairs, scores)

# Hierarchical agglomerative clustering
clusterer = HACClusterer(linkage='average', distance_threshold=0.5)
clusters = clusterer.cluster(pairs, scores)
```

### Evaluate Clustering

```python
from bench.metrics.clustering import compute_clustering_metrics

# True labels: {record_id: cluster_id}
true_labels = split.cluster_labels

# Predicted clusters: [[id1, id2, ...], [id3, id4, ...], ...]
pred_clusters = clusterer.cluster(pairs, scores)

metrics = compute_clustering_metrics(true_labels, pred_clusters)
print(f"ARI: {metrics['ari']:.4f}")
print(f"B³ F1: {metrics['b3_f1']:.4f}")
```

---

## 📄 Output Formats

### CSV Format

```csv
dataset,precision,recall,f1,accuracy,threshold,auc,ari_cc,ari_ag,p_at_k,r_at_k
ICE-ID,0.9954,0.9923,0.9938,0.9958,0.5,0.9984,0.881,0.4135,0.9949,0.9949
ABT-BUY,0.0696,0.9756,0.1299,0.09,0.0673,0.4915,0.0004,0.102,0.0244,0.0244
```

### JSON Format

```json
[
  {
    "dataset": "ICE-ID",
    "precision": 0.9954,
    "recall": 0.9923,
    "f1": 0.9938,
    "accuracy": 0.9958,
    "threshold": 0.5,
    "auc": 0.9984,
    "ari_cc": 0.881,
    "ari_ag": 0.4135,
    "p_at_k": 0.9949,
    "r_at_k": 0.9949
  }
]
```

---

## 🆘 Troubleshooting

### Out of Memory

```python
# Reduce sample size
dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

# Sample subset
sample_ids = split.records['id'].sample(n=10000, random_state=42).tolist()
split.records = split.records[split.records['id'].isin(sample_ids)]
```

### Slow Blocking

```python
# Reduce max_block_size
blocker = TokenBlocker(fields=['nafn_norm'], max_block_size=100)  # Smaller blocks
```

### Zero F1 on Small Datasets

Some datasets (Beer, iTunes-Amazon) have very few positives in test:
- **Beer**: 2 positives
- **iTunes-Amazon**: 3 positives

This makes metrics unreliable. Consider:
1. Using different splits
2. Reporting with confidence intervals
3. Noting limitations in paper

### External Model Errors

```bash
# Ensure external repos are cloned
cd bench/external
./setup_external.sh

# Check GPU availability (for Ditto, HierGAT)
python -c "import torch; print(torch.cuda.is_available())"
```

### Missing Dependencies

```bash
# Install all requirements
pip install -r requirements.txt
cd bench && pip install -r requirements.txt
cd external && pip install -r requirements_external.txt
```

---

## 📚 Related Documentation

- [WORKFLOWS.md](WORKFLOWS.md) — Step-by-step workflows
- [ARCHITECTURE.md](ARCHITECTURE.md) — Codebase structure
- [QUICK_START.md](../QUICK_START.md) — Getting started guide

---

**Need help?** Open an issue on GitHub or check the [workflows guide](WORKFLOWS.md).
