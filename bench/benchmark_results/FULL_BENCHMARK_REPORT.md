# Complete Entity Resolution Benchmark Report

**Generated**: 2026-01-09

## Executive Summary

This benchmark evaluates **10 models** across **9 datasets**:
- **ICE-ID** (Icelandic Historical Records from HuggingFace)
- **8 DeepMatcher datasets** (Product/Citation/Restaurant matching)

**Total experiments completed**: 89

## All Results Summary

| Model | Datasets | Avg F1 | Best F1 | Best Dataset | Status |
|-------|----------|--------|---------|--------------|--------|
| **ZeroER** | 8 | **0.647** | 0.991 | DBLP-ACM | ✅ Complete |
| **AnyMatch** | 9 | 0.729 | 1.000 | Fodors-Zagats | ✅ Complete |
| **Ditto** | 9 | 0.729 | 1.000 | ICE-ID | ✅ Complete |
| **MatchGPT** | 9 | 0.402 | 0.727 | DBLP-ACM | ✅ Complete |
| **HierGAT** | 9 | 0.382 | 1.000 | ICE-ID | ✅ Complete |
| **Fellegi-Sunter** | 9 | 0.332 | 0.950 | ICE-ID | ✅ Complete |
| **Rules** | 9 | 0.332 | 0.948 | ICE-ID | ✅ Complete |
| **NARS** | 9 | 0.301 | 0.667 | ICE-ID | ✅ Complete |
| **Gradient Boosting** | 9 | 0.247 | 0.745 | iTunes-Amazon | ✅ Complete |
| **Random Forest** | 9 | 0.117 | 0.565 | ICE-ID | ✅ Complete |

## Complete Results Matrix

### ICE-ID Dataset (Your Dataset - Icelandic Historical Records)

| Rank | Model | F1 | Precision | Recall | Time |
|------|-------|-----|-----------|--------|------|
| 1 | **Ditto** | **1.000** | 1.000 | 1.000 | 56.9s |
| 1 | **HierGAT** | **1.000** | 1.000 | 1.000 | 14.9s |
| 3 | Fellegi-Sunter | 0.950 | - | - | 2.7s |
| 4 | Rules | 0.948 | - | - | 0.8s |
| 5 | **AnyMatch** | **0.907** | - | - | ~600s |
| 6 | NARS | 0.667 | - | - | 2.9s |
| 7 | Gradient Boosting | 0.611 | - | - | 3.2s |
| 8 | Random Forest | 0.565 | - | - | 1.9s |
| 9 | MatchGPT | 0.276 | - | - | 26.5s |

### External Datasets - Complete Model Comparison (F1 Scores)

| Dataset | ZeroER | AnyMatch | Ditto | HierGAT | MatchGPT | F-S | Rules | NARS | GB | RF |
|---------|--------|----------|-------|---------|----------|-----|-------|------|-----|-----|
| **DBLP-ACM** | **0.991** | 0.947 | 0.970 | 0.326 | 0.727 | 0.304 | 0.304 | 0.304 | 0.174 | 0.000 |
| **Fodors-Zagats** | 0.980 | **1.000** | 0.977 | 0.330 | 0.667 | 0.209 | 0.209 | 0.209 | 0.000 | 0.000 |
| **DBLP-Scholar** | 0.817 | **0.970** | 0.940 | 0.347 | 0.400 | 0.314 | 0.314 | 0.314 | 0.196 | 0.011 |
| **Walmart-Amazon** | 0.624 | **0.632** | 0.556 | 0.235 | 0.333 | 0.172 | 0.172 | 0.172 | 0.116 | 0.010 |
| **iTunes-Amazon** | 0.579 | 0.490 | 0.595 | 0.397 | 0.435 | 0.397 | 0.397 | 0.397 | **0.745** | 0.462 |
| **Abt-Buy** | 0.420 | 0.400 | **0.525** | 0.333 | 0.333 | 0.194 | 0.194 | 0.194 | 0.077 | 0.000 |
| **Amazon-Google** | 0.393 | 0.625 | **0.732** | 0.203 | 0.000 | 0.185 | 0.185 | 0.185 | 0.040 | 0.000 |
| **Beer** | 0.373 | **0.849** | 0.267 | 0.267 | 0.444 | 0.267 | 0.267 | 0.267 | 0.261 | 0.000 |

## Model Rankings

### By Average F1 (All Datasets)

| Rank | Model | Avg F1 | Datasets | Type |
|------|-------|--------|----------|------|
| 1 | **AnyMatch** | 0.758 | 9 | Zero-shot Transfer |
| 2 | **Ditto** | 0.729 | 9 | Fine-tuned PLM |
| 3 | **ZeroER** | 0.647 | 8 | Unsupervised |
| 4 | **MatchGPT** | 0.402 | 9 | LLM Zero-shot |
| 5 | **HierGAT** | 0.382 | 9 | Graph Neural Net |
| 6 | **Fellegi-Sunter** | 0.332 | 9 | Probabilistic |
| 7 | **Rules** | 0.332 | 9 | Deterministic |
| 8 | **NARS** | 0.301 | 9 | Non-Axiomatic Reasoning |
| 9 | **Gradient Boosting** | 0.247 | 9 | ML Ensemble |
| 10 | **Random Forest** | 0.117 | 9 | ML Ensemble |

### ICE-ID Performance Ranking

| Rank | Model | F1 | Notes |
|------|-------|----|-------|
| 1 | **Ditto** | 1.000 | Perfect via fine-tuning |
| 1 | **HierGAT** | 1.000 | Perfect via fine-tuning |
| 3 | Fellegi-Sunter | 0.950 | Classical probabilistic |
| 4 | Rules | 0.948 | Deterministic |
| 5 | **AnyMatch** | 0.907 | Zero-shot transfer! |
| 6 | NARS | 0.667 | Non-axiomatic |
| 7 | Gradient Boosting | 0.611 | ML ensemble |
| 8 | Random Forest | 0.565 | ML ensemble |
| 9 | MatchGPT | 0.276 | GPT-3.5 zero-shot |

## Key Findings

### 1. ZeroER Excels on Citation Matching (Unsupervised!)
- **F1=0.991** on DBLP-ACM (near-perfect, no labels needed!)
- **F1=0.980** on Fodors-Zagats
- **F1=0.817** on DBLP-Scholar
- Average F1=0.647 across all datasets

### 2. AnyMatch Shows Best Zero-Shot Transfer
- **F1=1.0** on Fodors-Zagats (perfect!)
- **F1=0.97** on DBLP-Scholar
- **F1=0.91** on ICE-ID (strong cross-domain!)
- Highest average F1=0.758

### 3. Deep Learning Dominates ICE-ID
- **Ditto** and **HierGAT** both achieve **F1=1.0**
- Pre-trained language models excel on structured genealogical data

### 4. Unsupervised vs Supervised Trade-off
- **ZeroER** (unsupervised): Strong on clean, structured data (DBLP-ACM: 0.991)
- **AnyMatch/Ditto** (supervised): More consistent across domains

### 5. Classical Methods Still Competitive
- Fellegi-Sunter achieves **F1=0.95** on ICE-ID
- Beats LLM-based approaches on structured records

## Dataset-Specific Winners

| Dataset | Winner | F1 | Method Type |
|---------|--------|-----|-------------|
| **ICE-ID** | Ditto/HierGAT | 1.000 | Fine-tuned PLM |
| **DBLP-ACM** | ZeroER | 0.991 | Unsupervised |
| **Fodors-Zagats** | AnyMatch | 1.000 | Zero-shot |
| **DBLP-Scholar** | AnyMatch | 0.970 | Zero-shot |
| **Beer** | AnyMatch | 0.849 | Zero-shot |
| **Amazon-Google** | Ditto | 0.732 | Fine-tuned |
| **Walmart-Amazon** | AnyMatch | 0.632 | Zero-shot |
| **iTunes-Amazon** | Grad.Boost | 0.745 | ML Ensemble |
| **Abt-Buy** | Ditto | 0.525 | Fine-tuned |

## Model Details

### ZeroER (Unsupervised EM-based)
- **Architecture**: Gaussian Mixture Model with EM algorithm
- **Training**: No labeled data required!
- **Best for**: Clean, structured data with clear matching signals
- **Highlight**: F1=0.991 on DBLP-ACM without any supervision

### AnyMatch (Zero-Shot Transfer)
- **Architecture**: BERT-base with leave-one-out training
- **Training**: Trained on 8 datasets, tested on held-out
- **Best for**: Zero-shot transfer to new domains
- **Highlight**: F1=0.907 on ICE-ID via transfer learning

### Ditto (Fine-tuned PLM)
- **Architecture**: DistilBERT cross-encoder
- **Training**: Fine-tuned per dataset
- **Best for**: Maximum accuracy with labeled data

### HierGAT (Graph Neural Network)
- **Architecture**: Hierarchical Graph Attention Network
- **Training**: Fine-tuned with graph structure
- **Best for**: Structured entity matching

### MatchGPT (LLM Zero-Shot)
- **Architecture**: GPT-3.5-turbo via OpenAI API
- **Training**: Zero-shot with prompting
- **Best for**: Quick evaluation without training

## Reproduce This Benchmark

```bash
cd ICE-ID-2.0/bench

# 1. Install dependencies
pip install -r requirements.txt
pip install autogluon openai python-dotenv py_entitymatching

# 2. Set API keys
echo "GPT_KEY=your_key" > ../.env

# 3. Clone external repos
cd external && ./setup_external.sh && cd ..

# 4. Run internal models
python run_all_models_all_datasets.py

# 5. Run Ditto
python run_ditto_all.py
python run_missing.py  # Ditto on ICE-ID

# 6. Run ZeroER
python setup_zeroer_complete.py  # Prepare datasets
python run_zeroer_eval.py        # Run on all

# 7. Run external models
python run_external_models.py    # MatchGPT
python run_hiergat_all.py        # HierGAT

# 8. Run AnyMatch (GPU recommended)
cd external/anymatch
python loo.py --base_model bert-base --leaved_dataset_name iceid --train_data row
```

## Files Generated

| File | Experiments |
|------|-------------|
| `all_results.csv` | Internal models (45) |
| `ditto_results.csv` | Ditto (9) |
| `zeroer_results.csv` | ZeroER (8) |
| `hiergat_results.csv` | HierGAT (9) |
| `external_models_results.csv` | MatchGPT (9) |
| `anymatch_results.csv` | AnyMatch (9) |

## Conclusions

1. **ZeroER achieves F1=0.991 on DBLP-ACM without any labels** - best unsupervised result

2. **AnyMatch has highest average F1=0.758** via zero-shot transfer

3. **ICE-ID achieves perfect F1=1.0** with Ditto and HierGAT

4. **AnyMatch achieves F1=0.907 on ICE-ID** without training on it

5. **Model recommendations**:
   - Clean structured data, no labels → **ZeroER**
   - Zero-shot to new domains → **AnyMatch**
   - Maximum accuracy with fine-tuning → **Ditto/HierGAT**
   - Quick evaluation → **MatchGPT**
   - Structured records with rules → **Fellegi-Sunter**

## Complete F1 Results Table

| Dataset | ZeroER | AnyMatch | Ditto | HierGAT | MatchGPT | F-S | Rules | NARS | GB | RF |
|---------|--------|----------|-------|---------|----------|-----|-------|------|-----|-----|
| **ICE-ID** | - | 0.907 | 1.000 | 1.000 | 0.276 | 0.950 | 0.948 | 0.667 | 0.611 | 0.565 |
| **DBLP-ACM** | 0.991 | 0.947 | 0.970 | 0.326 | 0.727 | 0.304 | 0.304 | 0.304 | 0.174 | 0.000 |
| **Fodors-Zagats** | 0.980 | 1.000 | 0.977 | 0.330 | 0.667 | 0.209 | 0.209 | 0.209 | 0.000 | 0.000 |
| **DBLP-Scholar** | 0.817 | 0.970 | 0.940 | 0.347 | 0.400 | 0.314 | 0.314 | 0.314 | 0.196 | 0.011 |
| **Walmart-Amazon** | 0.624 | 0.632 | 0.556 | 0.235 | 0.333 | 0.172 | 0.172 | 0.172 | 0.116 | 0.010 |
| **iTunes-Amazon** | 0.579 | 0.490 | 0.595 | 0.397 | 0.435 | 0.397 | 0.397 | 0.397 | 0.745 | 0.462 |
| **Abt-Buy** | 0.420 | 0.400 | 0.525 | 0.333 | 0.333 | 0.194 | 0.194 | 0.194 | 0.077 | 0.000 |
| **Amazon-Google** | 0.393 | 0.625 | 0.732 | 0.203 | 0.000 | 0.185 | 0.185 | 0.185 | 0.040 | 0.000 |
| **Beer** | 0.373 | 0.849 | 0.267 | 0.267 | 0.444 | 0.267 | 0.267 | 0.267 | 0.261 | 0.000 |
| **Average** | 0.647 | 0.758 | 0.729 | 0.382 | 0.402 | 0.332 | 0.332 | 0.301 | 0.247 | 0.117 |
