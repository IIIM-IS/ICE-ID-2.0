# ICE-ID Workflows

Step-by-step guides for common tasks with the ICE-ID benchmarking framework.

---

## 📋 Table of Contents

1. [Running NARS Evaluation](#1-running-nars-evaluation)
2. [Evaluating External Models](#2-evaluating-external-models)
3. [Generating Paper Artifacts](#3-generating-paper-artifacts)
4. [Adding a New Dataset](#4-adding-a-new-dataset)
5. [Adding a New Model](#5-adding-a-new-model)
6. [Running Ablation Studies](#6-running-ablation-studies)
7. [Compiling Papers](#7-compiling-papers)
8. [Using the Dashboard](#8-using-the-dashboard)

---

## 1. Running NARS Evaluation

### Quick Run (All Datasets)

```bash
cd ICE-ID-2.0/bench
python scripts/run_nars_full_eval.py
```

This evaluates NARS on:
- ✅ ICE-ID (temporal split)
- ✅ All 8 DeepMatcher datasets (Abt-Buy, Amazon-Google, etc.)

**Output:**
- `benchmark_results/nars_full_eval.csv` — All metrics
- `paper_artifacts/nars_full_eval.json` — JSON version

### What It Does

```
1. Load ICE-ID dataset with temporal splits
   └── Train: pre-1870, Val: 1870-1890, Test: 1891-1920

2. Generate candidate pairs
   └── Sample positive pairs from clusters
   └── Sample hard negatives from same blocking partitions

3. Train NARS model
   └── Learn judgment weights from training pairs
   └── Calibrate threshold on validation

4. Evaluate on test set
   └── Pairwise: P, R, F1, Acc, AUC
   └── Ranking: P@k, R@k
   └── Clustering: ARI-CC, ARI-AG

5. Repeat for each DeepMatcher dataset
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **P (Precision)** | TP / (TP + FP) |
| **R (Recall)** | TP / (TP + FN) |
| **F1** | Harmonic mean of P and R |
| **Acc** | Overall accuracy |
| **AUC** | Area under ROC curve |
| **ARI-CC** | Adjusted Rand Index (connected components) |
| **ARI-AG** | Adjusted Rand Index (agglomerative) |
| **P@k** | Precision at k (k = #positives) |
| **R@k** | Recall at k |

---

## 2. Evaluating External Models

### Setup External Dependencies

```bash
cd ICE-ID-2.0/bench/external
./setup_external.sh
pip install -r requirements_external.txt
```

### Run Ditto

```python
from bench.models.ditto_adapter import DittoModel
from bench.data.iceid import IceIdDataset

dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

model = DittoModel()
model.fit(split, split.train_pairs, split.val_pairs)
scores = model.score(split, test_pairs)
```

### Run ZeroER (Unsupervised)

```python
from bench.models.zeroer_adapter import ZeroERModel
from bench.data.deepmatcher import DeepMatcherDataset

dataset = DeepMatcherDataset('dblp_acm')
split = dataset.load()

model = ZeroERModel()
# ZeroER doesn't need training - it's unsupervised
scores = model.score(split, candidate_pairs)
```

### Run MatchGPT (LLM-based)

```bash
export OPENAI_API_KEY="your-key-here"
```

```python
from bench.models.matchgpt_adapter import MatchGPTModel

model = MatchGPTModel(model_name='gpt-4')
# ... evaluation code
```

---

## 3. Generating Paper Artifacts

### Generate All Artifacts

```bash
cd ICE-ID-2.0/bench
python scripts/generate_paper_artifacts.py all
```

### Generate Specific Artifacts

```bash
# Dataset figures only
python scripts/generate_paper_artifacts.py figures

# Tables only
python scripts/generate_paper_artifacts.py tables

# External dataset comparison
python scripts/generate_paper_artifacts.py external
```

### What Gets Generated

**Figures (`paper_artifacts/plot_data/`):**
- `fig1_temporal_coverage.json` — Census wave distribution
- `fig2_missingness.json` — Missing data rates
- `fig3_cluster_sizes.json` — Cluster size CCDF
- `fig4_ambiguity.json` — Name collision analysis
- `fig5_blocking.json` — Blocking efficiency curves

**Tables (`paper_artifacts/table_data/`):**
- `table1_dataset_synopsis.csv` — Dataset comparison
- `table2_schema_matrix.csv` — Feature availability
- `table3_protocols_splits.csv` — Evaluation protocols

### Regenerate After Code Changes

If you modify the evaluation code:

```bash
# 1. Rerun evaluation
python scripts/run_nars_full_eval.py

# 2. Regenerate artifacts
python scripts/generate_paper_artifacts.py all

# 3. Recompile papers
cd ../papers
pdflatex main_nars_paper.tex
pdflatex main_data_paper.tex
```

---

## 4. Adding a New Dataset

### Step 1: Create Dataset Provider

Create `bench/bench/data/my_dataset.py`:

```python
from .base import BaseDataset
from ..core.types import DatasetSplit
from ..core.registry import get_registry

class MyDataset(BaseDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__("my_dataset", data_dir, **kwargs)
    
    def load(self) -> DatasetSplit:
        # Load your data
        records = pd.read_csv(self.data_dir / "records.csv")
        
        # Create splits
        train_pairs = ...
        val_pairs = ...
        test_pairs = ...
        
        return DatasetSplit(
            name="my_dataset",
            records=records,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            cluster_labels=cluster_labels,
        )
    
    def get_ground_truth(self):
        # Return list of true match pairs
        return positive_pairs, negative_pairs

# Register the dataset
get_registry("datasets").register("my_dataset", MyDataset)
```

### Step 2: Add to `__init__.py`

Edit `bench/bench/data/__init__.py`:

```python
from .my_dataset import MyDataset
```

### Step 3: Test Loading

```python
from bench.data.my_dataset import MyDataset

ds = MyDataset(data_dir="/path/to/data")
split = ds.load()
print(f"Records: {len(split.records)}")
print(f"Train pairs: {len(split.train_pairs)}")
```

---

## 5. Adding a New Model

### Step 1: Create Model Adapter

Create `bench/bench/models/my_model.py`:

```python
from .base import BaseModel
from ..core.types import DatasetSplit, Pair
from ..core.registry import get_registry
import numpy as np

class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__("my_model", **kwargs)
        self.threshold = 0.5
    
    def fit(
        self,
        dataset: DatasetSplit,
        train_pairs: pd.DataFrame,
        val_pairs: pd.DataFrame = None,
    ):
        """Train the model on labeled pairs."""
        # Your training logic here
        for _, row in train_pairs.iterrows():
            id1, id2, label = row['id1'], row['id2'], row['label']
            rec1 = dataset.get_record_by_id(id1)
            rec2 = dataset.get_record_by_id(id2)
            # Learn from (rec1, rec2, label)
        
        self._is_fitted = True
    
    def score(
        self,
        dataset: DatasetSplit,
        pairs: list,
    ) -> np.ndarray:
        """Score pairs and return probabilities."""
        scores = []
        for id1, id2 in pairs:
            rec1 = dataset.get_record_by_id(id1)
            rec2 = dataset.get_record_by_id(id2)
            # Compute similarity score
            score = self._compute_similarity(rec1, rec2)
            scores.append(score)
        return np.array(scores)
    
    def _compute_similarity(self, rec1, rec2) -> float:
        # Your similarity logic
        return 0.5

# Register
get_registry("models").register("my_model", MyModel)
```

### Step 2: Add to `__init__.py`

Edit `bench/bench/models/__init__.py`:

```python
from .my_model import MyModel
```

### Step 3: Evaluate

```python
from bench.data.iceid import IceIdDataset
from bench.models.my_model import MyModel
from bench.metrics.pairwise import compute_pairwise_metrics

# Load data
dataset = IceIdDataset()
split = dataset.load()

# Train
model = MyModel()
model.fit(split, split.train_pairs, split.val_pairs)

# Evaluate
test_pairs = [(row['id1'], row['id2']) for _, row in split.test_pairs.iterrows()]
test_labels = split.test_pairs['label'].values
scores = model.score(split, test_pairs)

metrics = compute_pairwise_metrics(test_labels, y_scores=scores, threshold=model.threshold)
print(f"F1: {metrics['f1']:.3f}")
```

---

## 6. Running Ablation Studies

### NARS Ablation (Judgment Types)

The NARS model supports ablation via `exclude_judgments`:

```python
from bench.models.nars import NarsModel

# Full model
model_full = NarsModel(preprocess="iceid")

# Remove name judgments
model_no_names = NarsModel(
    preprocess="iceid",
    exclude_judgments=["nafn", "first_name", "patronym", "surname"]
)

# Remove geographic judgments
model_no_geo = NarsModel(
    preprocess="iceid", 
    exclude_judgments=["farm", "parish", "district", "county"]
)
```

### Run Full Ablation Suite

```python
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel
from bench.metrics.pairwise import compute_pairwise_metrics
import pandas as pd

dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

ablations = [
    ('Full model', []),
    ('- Name judgments', ['nafn', 'first_name', 'patronym', 'surname']),
    ('- Birthyear judgments', ['birthyear']),
    ('- Geographic judgments', ['farm', 'parish', 'district', 'county']),
    ('- Sex judgments', ['sex']),
    ('- Census year (heimild)', ['heimild', 'census']),
]

results = []
for name, exclude in ablations:
    model = NarsModel(preprocess='iceid', exclude_judgments=exclude)
    model.fit(split, split.train_pairs, split.val_pairs)
    
    test_pairs = [(row['id1'], row['id2']) for _, row in split.test_pairs.iterrows()]
    scores = model.score(split, test_pairs)
    labels = split.test_pairs['label'].values
    
    metrics = compute_pairwise_metrics(labels, y_scores=scores, threshold=model.threshold)
    results.append({'ablation': name, 'f1': metrics['f1'], 'auc': metrics['auc']})

pd.DataFrame(results).to_csv('nars_ablations.csv', index=False)
```

---

## 7. Compiling Papers

### Prerequisites

```bash
# Install LaTeX (Ubuntu/Debian)
sudo apt install texlive-latex-extra texlive-fonts-recommended

# Install bibtex
sudo apt install bibtex
```

### Compile NARS Paper

```bash
cd ICE-ID-2.0/papers

# First pass
pdflatex main_nars_paper.tex

# Build bibliography
bibtex main_nars_paper

# Second pass (resolve references)
pdflatex main_nars_paper.tex

# Third pass (finalize)
pdflatex main_nars_paper.tex
```

### Compile Data Paper

```bash
cd ICE-ID-2.0/papers
pdflatex main_data_paper.tex
bibtex main_data_paper
pdflatex main_data_paper.tex
pdflatex main_data_paper.tex
```

### Quick Compile (No Bibliography)

```bash
pdflatex -interaction=nonstopmode main_nars_paper.tex
pdflatex -interaction=nonstopmode main_data_paper.tex
```

---

## 8. Using the Dashboard

### Start the Dashboard

```bash
cd ICE-ID-2.0
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

### Dashboard Features

1. **Dataset Inspector** — Browse records, view distributions
2. **Model Training** — Train models interactively
3. **Evaluation** — Run evaluations with visualizations
4. **Blocking Analysis** — Compare blocking strategies
5. **Settings** — Configure experiment parameters

### Example: Train and Evaluate

1. Select dataset (ICE-ID or DeepMatcher)
2. Choose model (NARS, Fellegi-Sunter, etc.)
3. Configure hyperparameters
4. Click "Train"
5. View metrics and confusion matrix
6. Export results

---

## 🔄 Common Command Patterns

### Full Evaluation Pipeline

```bash
cd ICE-ID-2.0/bench

# 1. Prepare data
python scripts/prepare_data.py

# 2. Run NARS evaluation
python scripts/run_nars_full_eval.py

# 3. Generate artifacts
python scripts/generate_paper_artifacts.py all

# 4. Compile papers
cd ../papers
pdflatex main_nars_paper.tex
pdflatex main_data_paper.tex
```

### Quick Smoke Test

```bash
cd ICE-ID-2.0/bench
python -c "
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel

ds = IceIdDataset(data_dir='../raw_data')
split = ds.load()
print(f'Loaded {len(split.records)} records')

model = NarsModel(preprocess='iceid')
print('Model initialized successfully')
"
```

### Check Artifact Consistency

```bash
cd ICE-ID-2.0/bench
python -c "
import json
import pandas as pd

# Load CSV
df = pd.read_csv('benchmark_results/nars_full_eval.csv')

# Load JSON
with open('paper_artifacts/nars_full_eval.json') as f:
    data = json.load(f)

# Compare
assert len(df) == len(data), 'Mismatch!'
print('Artifacts are consistent')
"
```

---

## 📚 Related Documentation

- [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) — Detailed benchmarking guide
- [ARCHITECTURE.md](ARCHITECTURE.md) — Codebase structure
- [QUICK_START.md](../QUICK_START.md) — Getting started guide

---

**Need help?** Check the [troubleshooting section](BENCHMARK_GUIDE.md#troubleshooting) or open an issue on GitHub.
