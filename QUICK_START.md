# Quick Start Guide

Get up and running with ICE-ID in 5 minutes.

---

## 🚀 Installation

### Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/ice-id.git
cd ice-id/ICE-ID-2.0

# Install dependencies
pip install -r requirements.txt

# Install benchmark dependencies
cd bench
pip install -r requirements.txt
```

### Step 2: Get the Data

```bash
# Clone ICE-ID dataset from HuggingFace
git lfs install
git clone https://huggingface.co/datasets/goldpotatoes/ice-id ../raw_data/
```

---

## 🎯 Your First Benchmark

### Run NARS on All Datasets

```bash
cd bench
python scripts/run_nars_full_eval.py
```

This will:
- ✅ Load ICE-ID with temporal splits
- ✅ Evaluate NARS on ICE-ID and 8 DeepMatcher datasets
- ✅ Generate metrics (F1, AUC, ARI, P@k)
- ✅ Save results to `benchmark_results/nars_full_eval.csv`

**Expected output:**
```
Evaluating NARS on ICE-ID...
  F1: 0.994, AUC: 0.998, ARI-CC: 0.881

Evaluating NARS on Abt-Buy...
  F1: 0.130, AUC: 0.492

...
```

---

## 🐍 Python API Example

### Basic Usage

```python
from bench.data.iceid import IceIdDataset
from bench.models.nars import NarsModel
from bench.metrics.pairwise import compute_pairwise_metrics

# Load dataset
dataset = IceIdDataset(data_dir='../raw_data')
split = dataset.load()

# Train model
model = NarsModel(preprocess='iceid')
model.fit(split, split.train_pairs, split.val_pairs)

# Evaluate
test_pairs = [(row['id1'], row['id2']) for _, row in split.test_pairs.iterrows()]
test_labels = split.test_pairs['label'].values
scores = model.score(split, test_pairs)

# Compute metrics
metrics = compute_pairwise_metrics(
    test_labels, 
    y_scores=scores, 
    threshold=model.threshold
)

print(f"F1: {metrics['f1']:.3f}")
print(f"AUC: {metrics['auc']:.3f}")
```

---

## 🖥️ Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

**Features:**
- 📊 Browse and visualize datasets
- 🎯 Train models interactively
- 📈 View evaluation metrics and confusion matrices
- 🔍 Inspect clusters and predictions

---

## 📊 Available Datasets

### ICE-ID (Main Dataset)
- **984,028 records** from 1703–1920
- **226,864 unique persons**
- Temporal splits: Train (pre-1870), Val (1870–1890), Test (1891–1920)

### DeepMatcher Datasets
- **Abt-Buy** (Products)
- **Amazon-Google** (Products)
- **DBLP-ACM** (Citations)
- **DBLP-Scholar** (Citations)
- **Walmart-Amazon** (Products)
- **iTunes-Amazon** (Music)
- **Beer** (Beverages)
- **Fodors-Zagats** (Restaurants)

---

## 🤖 Available Models

### Internal Models
- **NARS** — Non-Axiomatic Reasoning System
- **Fellegi-Sunter** — Classic probabilistic linkage
- **Rules** — Rule-based matching
- **XGBoost** — XGBoost gradient boosting
- **LightGBM** — LightGBM gradient boosting
- **Random Forest** — Random forest ensemble
- **Gradient Boosting** — Scikit-learn gradient boosting

### External Models
- **Ditto** — Deep learning (VLDB 2021)
- **HierGAT** — Graph neural network (SIGMOD 2023)
- **ZeroER** — Unsupervised (VLDB 2022)
- **AnyMatch** — Zero-shot (VLDB 2024)
- **MatchGPT** — LLM-based (arXiv 2023)

---

## 📈 Metrics

**Pairwise:**
- Precision, Recall, F1, Accuracy, AUC

**Clustering:**
- ARI (Adjusted Rand Index)
- B³ F1

**Ranking:**
- P@k, R@k

---

## 🔧 Configuration

### YAML Config Example

Create `config.yaml`:

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

## 📚 Next Steps

1. **Read the documentation:**
   - [BENCHMARK_GUIDE.md](docs/BENCHMARK_GUIDE.md) — Detailed benchmarking guide
   - [WORKFLOWS.md](docs/WORKFLOWS.md) — Common workflows and tasks
   - [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Codebase structure

2. **Run your first experiment:**
   ```bash
   cd bench
   python scripts/run_nars_full_eval.py
   ```

3. **Explore the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

4. **Add your own model:**
   See [WORKFLOWS.md](docs/WORKFLOWS.md#5-adding-a-new-model) for instructions.

---

## 🆘 Troubleshooting

### Out of Memory
```python
# Sample subset of records
sample_ids = split.records['id'].sample(n=10000).tolist()
split.records = split.records[split.records['id'].isin(sample_ids)]
```

### Missing Dependencies
```bash
pip install -r requirements.txt
cd bench && pip install -r requirements.txt
```

### External Models Not Working
```bash
cd bench/external
./setup_external.sh
pip install -r requirements_external.txt
```

---

## 📜 Citation

If you use ICE-ID, please cite:

```bibtex
@article{iceid2026,
  title={ICE-ID: A Benchmark for Longitudinal Entity Resolution in Historical Census Data},
  author={Hora de Carvalho, Gonçalo and Popov, Lazar S. and Kaatee, Sander and 
          Thórisson, Kristinn R. and Li, Tangrui and Björnsson, Pétur Húni and 
          Dibangoye, Jilles S.},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  note={Datasets and Benchmarks Track}
}
```

---

**Ready to go?** Check out the [full documentation](docs/) or start with the [benchmarking guide](docs/BENCHMARK_GUIDE.md)!
