# ICE-ID Project Architecture

Complete map of every file and folder in the ICE-ID project, explaining what each component does and how they relate to each other.

---

## 📋 Table of Contents

- [Top-Level Structure](#top-level-structure)
- [Benchmarking Framework](#bench-benchmarking-framework)
- [Dashboard](#dashboard-interactive-web-dashboard)
- [Papers](#papers-latex-papers)
- [Data Storage](#data-data-storage)
- [Data Flow Overview](#data-flow-overview)
- [Module Dependencies](#module-dependencies)
- [Key Design Patterns](#key-design-patterns)

---

## Top-Level Structure

```
ICE-ID-2.0/
├── bench/                  # Main benchmarking framework
├── dashboard/              # Interactive web dashboard (Streamlit)
├── data/                   # Data storage (runs, versions, external)
├── docs/                   # Documentation
├── papers/                 # LaTeX papers and figures
├── raw_data/               # Source ICE-ID dataset files
├── runs/                   # Experiment run outputs
├── README.md               # Project overview
├── QUICK_START.md          # Getting started guide
└── requirements.txt        # Python dependencies
```

---

## `/bench/` — Benchmarking Framework

The core entity resolution benchmarking system. This is the primary codebase for running experiments.

### `/bench/bench/` — Python Package

```
bench/bench/
├── __init__.py             # Package initialization
├── cli.py                  # Command-line interface entry point
│
├── core/                   # Core types and utilities
│   ├── types.py            # DatasetSplit, Pair, Record, ClusterResult, etc.
│   ├── registry.py         # Plugin registry for models, datasets, blockers
│   └── random.py           # Reproducible random seed management
│
├── data/                   # Dataset providers
│   ├── base.py             # BaseDataset abstract class
│   ├── iceid.py            # ICE-ID dataset loader with temporal splits
│   ├── deepmatcher.py      # DeepMatcher datasets (Abt-Buy, DBLP-ACM, etc.)
│   ├── zenodo.py           # Zenodo-hosted datasets
│   ├── wdc_products.py      # WDC Products dataset
│   ├── additional_datasets.py  # FEBRL, Synthea, etc.
│   └── external_profiles.py    # Profiling utilities for external datasets
│
├── blocking/               # Candidate generation strategies
│   ├── base.py             # BaseBlocker abstract class
│   ├── token_blocking.py   # Token-based blocking (field tokenization)
│   ├── phonetic_blocking.py # Soundex/Metaphone blocking
│   └── geo_hierarchy.py    # Geographic hierarchy blocking
│
├── models/                 # Entity resolution models
│   ├── base.py             # BaseModel abstract class (fit, score, predict)
│   ├── nars.py             # NARS (Non-Axiomatic Reasoning System)
│   ├── fellegi_sunter.py   # Fellegi-Sunter probabilistic linkage
│   ├── rules.py            # Rule-based deterministic matcher
│   ├── ensemble.py         # ML ensemble models (XGBoost, LightGBM, RandomForest, GradientBoosting)
│   ├── ditto_adapter.py    # Ditto deep learning model adapter
│   ├── hiergat_adapter.py  # HierGAT graph attention network adapter
│   ├── zeroer_adapter.py   # ZeroER unsupervised matcher adapter
│   ├── anymatch_adapter.py # AnyMatch zero-shot adapter
│   ├── matchgpt_adapter.py # MatchGPT LLM-based matcher adapter
│   └── opennars_adapter.py # OpenNARS-for-Applications adapter
│
├── calibration/            # Score calibration methods
│   ├── base.py             # BaseCalibrator abstract class
│   ├── fixed_threshold.py  # Fixed threshold calibration
│   ├── platt.py            # Platt scaling (sigmoid fit)
│   └── isotonic.py         # Isotonic regression calibration
│
├── clustering/             # Entity clustering algorithms
│   ├── base.py             # BaseClusterer abstract class
│   ├── connected_components.py  # Graph connected components
│   └── hac.py              # Hierarchical agglomerative clustering
│
├── metrics/                # Evaluation metrics
│   ├── pairwise.py         # Precision, Recall, F1, AUC, AP
│   ├── ranking.py          # P@k, R@k
│   ├── clustering.py      # ARI, B³ F1
│   └── sanity.py           # Sanity checks (random baseline comparison)
│
├── runner/                 # Experiment execution
│   ├── run_one.py          # Single experiment runner
│   └── run_grid.py         # Grid search over configurations
│
├── config/                 # Configuration schemas
│   ├── schema.py           # Pydantic config validation
│   └── examples/           # Example YAML configs
│       ├── iceid_nars.yaml
│       ├── iceid_fellegi_sunter.yaml
│       └── zenodo_nars.yaml
│
└── pairs/                  # (Reserved for pair builders)
```

### `/bench/scripts/` — Executable Scripts

```
bench/scripts/
├── run_experiments.py      # Main experiment runner with subcommands
├── run_nars_full_eval.py   # NARS evaluation across all datasets
├── generate_paper_artifacts.py  # Generate figures/tables for papers
├── prepare_data.py         # Data preparation and download
└── fetch_external_datasets.py   # Download FEBRL, Synthea, ORCID, etc.
```

### `/bench/benchmark_results/` — Experiment Outputs

```
bench/benchmark_results/
├── nars_full_eval.csv      # NARS results on all datasets (Table 6)
├── nars_ablations.csv      # NARS ablation study results
├── nars_calibration_sensitivity.csv  # Calibration strategy comparison
├── nars_graph_eval.csv     # End-to-end graph evaluation
├── ditto_results.csv       # Ditto model results
├── hiergat_results.csv     # HierGAT model results
├── zeroer_results.csv      # ZeroER model results
├── anymatch_results.csv    # AnyMatch model results
└── FULL_BENCHMARK_REPORT.md # Summary report
```

### `/bench/paper_artifacts/` — Paper-Ready Data

```
bench/paper_artifacts/
├── plot_data/              # JSON/CSV for figures
│   ├── fig1_temporal_coverage.json
│   ├── fig2_missingness.json
│   ├── fig3_cluster_sizes.json
│   ├── fig4_ambiguity.json
│   ├── fig5_blocking.json
│   ├── calibration_sensitivity.json
│   └── nars_rerun_f1.json
│
├── table_data/             # JSON/CSV for tables
│   ├── table1_dataset_synopsis.csv
│   ├── table2_schema_matrix.csv
│   ├── table3_protocols_splits.csv
│   ├── table_longitudinal_comparison.json
│   └── table_external_datasets.json
│
├── nars_full_eval.json     # NARS results (JSON format)
├── nars_graph_eval.json    # Graph evaluation results
└── nars_calibration_sensitivity.json
```

### `/bench/deepmatcher_data/` — Classic ER Datasets

```
bench/deepmatcher_data/
├── abt_buy/                # Abt-Buy product matching
├── amazon_google/         # Amazon-Google product matching
├── dblp_acm/               # DBLP-ACM citation matching
├── dblp_scholar/           # DBLP-Google Scholar citation matching
├── itunes_amazon/          # iTunes-Amazon music matching
├── walmart_amazon/         # Walmart-Amazon product matching
├── beer/                   # BeerAdvocate-RateBeer matching
└── fodors_zagats/          # Fodors-Zagats restaurant matching
```

Each dataset folder contains:
- `tableA.csv`, `tableB.csv` — Source tables
- `train.csv`, `valid.csv`, `test.csv` — Labeled pairs

### `/bench/external/` — External Model Repositories

```
bench/external/
├── ditto/                  # Ditto deep learning model (cloned repo)
├── zeroer/                 # ZeroER unsupervised model
├── anymatch/               # AnyMatch zero-shot model
├── MatchGPT/               # MatchGPT LLM-based model
├── OpenNARS-for-Applications/  # OpenNARS C implementation
├── wdcproducts/             # WDC Products dataset tools
├── requirements_external.txt   # Dependencies for external models
└── setup_external.sh       # Setup script for external repos
```

---

## `/dashboard/` — Interactive Web Dashboard

A Streamlit-based dashboard for interactive exploration and evaluation.

```
dashboard/
├── app.py                  # Main Streamlit application entry
├── backends.py             # Backend service connections
├── er_bench.py             # Benchmark interface
├── eval_api.py             # Evaluation API endpoints
├── train_api.py            # Training API endpoints
├── graphing.py             # Visualization utilities
├── inspector_tab.py        # Data inspection UI
├── model_registry.py       # Model management
├── settings_manager.py     # Configuration management
├── schemas.py              # Data schemas
├── ds_io.py                # Dataset I/O utilities
├── edits.py                # Data editing utilities
├── external_models.py      # External model integration
│
├── blocking/               # Blocking UI components
├── calibration/            # Calibration UI components
├── clustering/             # Clustering UI components
├── datasets/               # Dataset UI components
├── metrics/                # Metrics display components
├── models/                 # Model UI components
└── tests/                  # Dashboard tests
```

---

## `/papers/` — LaTeX Papers

```
papers/
├── main_data_paper.tex     # Dataset paper (ICE-ID description)
├── main_nars_paper.tex     # Methods paper (NARS evaluation)
├── main.bib                # BibTeX references
├── neurips_2024.sty        # NeurIPS style file
├── notes.txt               # Author notes
│
├── figures/                # Generated figures
│   ├── fig1_temporal_coverage.pdf
│   ├── fig2_missingness.pdf
│   ├── fig3_cluster_sizes.pdf
│   ├── fig4_ambiguity.pdf
│   ├── fig5_blocking.pdf
│   ├── nars_rerun_f1.pdf
│   ├── calibration_sensitivity.png
│   ├── cross_dataset_heatmap.png
│   ├── ablation_chart.png
│   └── missingno_iceid.pdf
│
└── *.pdf                   # Compiled papers
```

---

## `/raw_data/` — ICE-ID Source Data

The original Icelandic census data files:

```
raw_data/
├── people.csv              # 984,028 census records (main table)
├── counties.csv            # County geographic hierarchy
├── districts.csv           # District geographic hierarchy
├── parishes.csv            # Parish geographic hierarchy
└── manntol_einstaklingar_new.csv  # Expert-curated person labels
```

### Key Fields in `people.csv`:

| Field | Description |
|-------|-------------|
| `id` | Unique record identifier |
| `person` | Cluster label (same person across censuses) |
| `heimild` | Census year (1703, 1801, ..., 1920) |
| `nafn_norm` | Normalized full name |
| `first_name`, `patronym`, `surname` | Name components |
| `birthyear`, `sex`, `marriagestatus` | Demographics |
| `farm`, `parish`, `district`, `county` | 4-level geography |
| `partner`, `father`, `mother` | Kinship links |

---

## `/data/` — Data Storage

```
data/
├── raw_data/               # Symlink or copy of /raw_data/
├── runs/                   # Dashboard experiment runs
├── versions/               # Dataset versioning
└── external_datasets/      # Downloaded external datasets
    ├── febrl/              # FEBRL synthetic data
    ├── synthea/            # Synthea synthetic patients
    ├── orcid/              # ORCID researcher data
    ├── semparl/            # SemParl parliamentary data
    ├── ckcc/               # CKCC correspondence data
    └── correspsearch/      # correspSearch correspondence
```

---

## `/runs/` — Experiment Outputs

```
runs/
├── er_bench_full/          # Full benchmark runs
├── external_models/        # External model evaluation results
├── test_all_models/        # Comprehensive model tests
├── test_single/            # Single-model test runs
├── hundred_loose_dual/     # Specific experiment configurations
└── settings/               # Saved experiment settings
```

---

## Data Flow Overview

```
                    ┌──────────────┐
                    │   raw_data/  │
                    │  people.csv  │
                    └──────┬───────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  bench/bench/data/    │
               │  iceid.py loads data  │
               │  + temporal splits    │
               └───────────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌──────────┐   ┌──────────┐   ┌──────────┐
     │ blocking │   │  models  │   │ metrics  │
     │  tokens  │   │   NARS   │   │   F1     │
     │   geo    │   │  Ditto   │   │   ARI    │
     └────┬─────┘   └────┬─────┘   └────┬─────┘
          │              │              │
          └──────────────┼──────────────┘
                         ▼
               ┌───────────────────────┐
               │  scripts/run_*.py     │
               │  Execute experiments  │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  benchmark_results/   │
               │  CSV/JSON outputs     │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  paper_artifacts/     │
               │  Figures & Tables     │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  papers/*.tex         │
               │  LaTeX compilation    │
               └───────────────────────┘
```

---

## Module Dependencies

```
core/types.py          ← Used by everything
       │
       ├── data/base.py → iceid.py, deepmatcher.py, ...
       │
       ├── blocking/base.py → token_blocking.py, phonetic_blocking.py
       │
       ├── models/base.py → nars.py, fellegi_sunter.py, ditto_adapter.py
       │
       ├── calibration/base.py → platt.py, isotonic.py
       │
       ├── clustering/base.py → connected_components.py, hac.py
       │
       └── metrics/ → pairwise.py, ranking.py, clustering.py
              │
              └── runner/run_one.py → orchestrates all above
                     │
                     └── scripts/run_experiments.py → CLI entry
```

---

## Key Design Patterns

### 1. Plugin Registry

All models, datasets, and blockers register themselves:

```python
from bench.core.registry import get_registry
get_registry("models").register("nars", NarsModel)
```

### 2. DatasetSplit Container

A unified container for both deduplication and two-table ER:

```python
@dataclass
class DatasetSplit:
    name: str
    records: Optional[pd.DataFrame]      # For dedup
    left_table: Optional[pd.DataFrame]   # For two-table
    right_table: Optional[pd.DataFrame]
    train_pairs: Optional[pd.DataFrame]
    val_pairs: Optional[pd.DataFrame]
    test_pairs: Optional[pd.DataFrame]
    cluster_labels: Optional[Dict[int, int]]
```

### 3. Temporal OOD Splits

ICE-ID uses strictly temporal splits to simulate real deployment:
- **Train**: pre-1870
- **Validation**: 1870–1890
- **Test**: 1891–1920

### 4. Artifact-Backed Reporting

All paper figures/tables are generated from JSON/CSV artifacts:
- Scripts write to `paper_artifacts/`
- LaTeX references these files
- Ensures reproducibility and consistency

---

## 📚 Related Documentation

- [WORKFLOWS.md](WORKFLOWS.md) — Step-by-step workflows
- [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) — Detailed benchmarking guide
- [QUICK_START.md](../QUICK_START.md) — Getting started guide

---

**Questions?** Check the [workflows guide](WORKFLOWS.md) or open an issue on GitHub.
