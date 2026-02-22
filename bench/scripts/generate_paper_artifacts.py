#!/usr/bin/env python3
"""
Consolidated script for generating all paper figures and tables.

Usage:
    python scripts/generate_paper_artifacts.py all          # Generate everything
    python scripts/generate_paper_artifacts.py figures      # Dataset figures only
    python scripts/generate_paper_artifacts.py tables       # Dataset tables only
    python scripts/generate_paper_artifacts.py nars         # NARS-specific figures/tables
    python scripts/generate_paper_artifacts.py external     # External dataset comparison
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List

# Paths
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
RAW_DATA = BASE_DIR.parent / "raw_data"
FIGURES_DIR = BASE_DIR.parent / "papers" / "figures"
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts"
PLOT_DATA_DIR = ARTIFACTS_DIR / "plot_data"
TABLE_DATA_DIR = ARTIFACTS_DIR / "table_data"
RESULTS_DIR = BASE_DIR / "results"
DEEPMATCHER_DATA = BASE_DIR / "deepmatcher_data"
EXTERNAL_DATA = BASE_DIR.parent / "data" / "external_datasets"

CHUNK_SIZE = 50000

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


def ensure_dirs():
    for d in [FIGURES_DIR, PLOT_DATA_DIR, TABLE_DATA_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATASET FIGURES (for main_data_paper.tex)
# =============================================================================

def compute_temporal_coverage() -> Dict[str, Any]:
    """Compute records per census wave and label density via chunked reads."""
    people_csv = RAW_DATA / "people.csv"
    
    wave_counts = defaultdict(int)
    wave_labeled = defaultdict(int)
    unique_persons = set()
    
    for chunk in pd.read_csv(people_csv, chunksize=CHUNK_SIZE, low_memory=False):
        for _, row in chunk.iterrows():
            heimild = row.get("heimild")
            person_id = row.get("person")
            
            if pd.notna(heimild):
                year = int(heimild)
                wave_counts[year] += 1
                if pd.notna(person_id):
                    wave_labeled[year] += 1
                    unique_persons.add(int(person_id))
    
    waves = sorted(wave_counts.keys())
    artifact = {
        "waves": waves,
        "records_per_wave": {str(w): wave_counts[w] for w in waves},
        "labeled_per_wave": {str(w): wave_labeled[w] for w in waves},
        "label_rate_per_wave": {str(w): round(wave_labeled[w] / wave_counts[w] * 100, 2) if wave_counts[w] > 0 else 0 for w in waves},
        "total_records": sum(wave_counts.values()),
        "total_labeled": sum(wave_labeled.values()),
        "unique_persons": len(unique_persons),
        "avg_label_rate": round(sum(wave_labeled.values()) / sum(wave_counts.values()) * 100, 2),
    }
    
    # Add classic dataset density for comparison
    classic_datasets = {}
    for ds_name in ["abt_buy", "amazon_google", "dblp_acm", "dblp_scholar", "walmart_amazon", "itunes_amazon", "beer", "fodors_zagats"]:
        ds_path = DEEPMATCHER_DATA / ds_name
        if ds_path.exists():
            try:
                a = pd.read_csv(ds_path / "tableA.csv", low_memory=False)
                b = pd.read_csv(ds_path / "tableB.csv", low_memory=False)
                classic_datasets[ds_name] = {
                    "n_records_a": len(a),
                    "n_records_b": len(b),
                    "n_total": len(a) + len(b),
                    "label_rate": 100.0,
                }
            except Exception:
                pass
    artifact["classic_label_density"] = classic_datasets
    
    return artifact


def figure1_temporal_coverage(artifact: Dict[str, Any]):
    """Figure 1: Temporal coverage + label density."""
    waves = artifact["waves"]
    records = [artifact["records_per_wave"][str(w)] for w in waves]
    label_rates = [artifact["label_rate_per_wave"][str(w)] for w in waves]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.bar(waves, records, color='steelblue', alpha=0.8, width=6)
    ax1.set_xlabel('Census Year')
    ax1.set_ylabel('Number of Records', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, max(records) * 1.15)
    
    ax1.axvline(x=1870, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(1870, max(records) * 1.05, 'Train/Test Split', ha='center', fontsize=9, color='red')
    
    ax2 = ax1.twinx()
    ax2.plot(waves, label_rates, 'o-', color='darkorange', linewidth=2, markersize=6)
    ax2.set_ylabel('Label Rate (%)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 100)
    
    plt.title(f'ICE-ID: Temporal Coverage and Label Density\n({artifact["total_records"]:,} records, {artifact["unique_persons"]:,} unique persons)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_temporal_coverage.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig1_temporal_coverage.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(PLOT_DATA_DIR / "fig1_temporal_coverage.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    df = pd.DataFrame({
        "wave": waves,
        "records": records,
        "labeled": [artifact["labeled_per_wave"][str(w)] for w in waves],
        "label_rate": label_rates,
    })
    df.to_csv(PLOT_DATA_DIR / "fig1_temporal_coverage.csv", index=False)
    
    print(f"  Figure 1: ICE-ID waves={len(waves)}, classic_datasets={len(artifact.get('classic_label_density', {}))}")


def compute_missingness() -> Dict[str, Any]:
    """Compute missingness rates by feature family per census wave."""
    people_csv = RAW_DATA / "people.csv"
    
    feature_families = {
        "names": ["nafn_norm", "first_name", "patronym", "surname"],
        "demographics": ["birthyear", "sex", "status", "marriagestatus"],
        "geography": ["farm", "parish", "district", "county"],
        "kinship": ["partner", "father", "mother"],
    }
    
    wave_counts = defaultdict(int)
    wave_missing = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for chunk in pd.read_csv(people_csv, chunksize=CHUNK_SIZE, low_memory=False):
        for _, row in chunk.iterrows():
            heimild = row.get("heimild")
            if pd.isna(heimild):
                continue
            year = int(heimild)
            wave_counts[year] += 1
            
            for family, cols in feature_families.items():
                for col in cols:
                    if pd.isna(row.get(col)) or str(row.get(col, "")).strip() == "":
                        wave_missing[year][family][col] += 1
    
    waves = sorted(wave_counts.keys())
    artifact = {"waves": waves, "feature_families": {}}
    
    for family, cols in feature_families.items():
        artifact["feature_families"][family] = {
            "columns": cols,
            "missing_rate_per_wave": {},
            "overall_missing_rate": 0,
        }
        total_missing = 0
        total_possible = 0
        for wave in waves:
            family_missing = sum(wave_missing[wave][family][col] for col in cols)
            family_possible = wave_counts[wave] * len(cols)
            rate = round(family_missing / family_possible * 100, 2) if family_possible > 0 else 0
            artifact["feature_families"][family]["missing_rate_per_wave"][str(wave)] = rate
            total_missing += family_missing
            total_possible += family_possible
        artifact["feature_families"][family]["overall_missing_rate"] = round(total_missing / total_possible * 100, 2) if total_possible > 0 else 0
    
    return artifact


def figure2_missingness(artifact: Dict[str, Any]):
    """Figure 2: Missingness over time by feature family."""
    waves = artifact["waves"]
    families = ["names", "demographics", "geography", "kinship"]
    colors = ["forestgreen", "steelblue", "darkorange", "indianred"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for family, color in zip(families, colors):
        rates = [artifact["feature_families"][family]["missing_rate_per_wave"][str(w)] for w in waves]
        ax.plot(waves, rates, 'o-', color=color, linewidth=2, markersize=5, 
                label=f'{family.capitalize()} ({artifact["feature_families"][family]["overall_missing_rate"]:.1f}%)')
    
    ax.set_xlabel('Census Year')
    ax.set_ylabel('Missing Rate (%)')
    ax.set_title('ICE-ID: Missingness by Feature Family Over Time')
    ax.legend(loc='upper left')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_missingness.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig2_missingness.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(PLOT_DATA_DIR / "fig2_missingness.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    print(f"  Figure 2: Names {artifact['feature_families']['names']['overall_missing_rate']}%, Kinship {artifact['feature_families']['kinship']['overall_missing_rate']}%")


def compute_cluster_sizes() -> Dict[str, Any]:
    """Compute cluster size distribution (records per person)."""
    people_csv = RAW_DATA / "people.csv"
    
    person_counts = Counter()
    
    for chunk in pd.read_csv(people_csv, chunksize=CHUNK_SIZE, low_memory=False):
        for _, row in chunk.iterrows():
            person_id = row.get("person")
            if pd.notna(person_id):
                person_counts[int(person_id)] += 1
    
    size_distribution = Counter(person_counts.values())
    sizes = sorted(size_distribution.keys())
    
    counts_list = list(person_counts.values())
    
    # Compute CCDF
    sorted_sizes = np.sort(counts_list)
    ccdf_x = np.unique(sorted_sizes)
    ccdf_y = np.array([np.mean(sorted_sizes >= x) for x in ccdf_x])
    
    artifact = {
        "cluster_size_counts": {str(s): size_distribution[s] for s in sizes},
        "total_persons": len(person_counts),
        "median_size": int(np.median(counts_list)) if counts_list else 0,
        "p95_size": int(np.percentile(counts_list, 95)) if counts_list else 0,
        "max_size": max(sizes) if sizes else 0,
        "persons_with_multiple_records": sum(1 for c in counts_list if c > 1),
        "ccdf_points": [{"size": int(x), "prob": float(y)} for x, y in zip(ccdf_x[:50], ccdf_y[:50])],
    }
    
    return artifact


def figure3_cluster_sizes(artifact: Dict[str, Any]):
    """Figure 3: Cluster size distribution (log-log CCDF)."""
    ccdf_points = artifact["ccdf_points"]
    sizes = [p["size"] for p in ccdf_points]
    probs = [p["prob"] for p in ccdf_points]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.loglog(sizes, probs, 'o-', color='steelblue', markersize=5)
    ax.set_xlabel('Cluster Size (records per person)')
    ax.set_ylabel('P(Size ≥ x)')
    ax.set_title(f'ICE-ID: Cluster Size CCDF\n(Median: {artifact["median_size"]}, 95th pctl: {artifact["p95_size"]}, Max: {artifact["max_size"]})')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_cluster_sizes.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig3_cluster_sizes.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(PLOT_DATA_DIR / "fig3_cluster_sizes.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    print(f"  Figure 3: {artifact['total_persons']:,} persons, median={artifact['median_size']}, max={artifact['max_size']}")


def compute_name_ambiguity() -> Dict[str, Any]:
    """Compute name frequency distribution and entropy."""
    people_csv = RAW_DATA / "people.csv"
    
    name_counts = Counter()
    total_records = 0
    
    for chunk in pd.read_csv(people_csv, chunksize=CHUNK_SIZE, low_memory=False):
        for _, row in chunk.iterrows():
            name = str(row.get("nafn_norm", "")).strip().lower()
            if name:
                name_counts[name] += 1
                total_records += 1
    
    top_names = name_counts.most_common(100)
    
    probs = np.array(list(name_counts.values())) / total_records
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    effective_names = 2 ** entropy
    
    top_10_share = sum(c for _, c in top_names[:10]) / total_records * 100
    
    artifact = {
        "top_100_names": [{"name": n, "count": c} for n, c in top_names],
        "total_unique_names": len(name_counts),
        "total_records": total_records,
        "entropy_bits": round(entropy, 2),
        "effective_names": round(effective_names, 0),
        "top_10_share_pct": round(top_10_share, 2),
    }
    
    return artifact


def figure4_ambiguity(artifact: Dict[str, Any]):
    """Figure 4: Name ambiguity (Zipf + entropy)."""
    top_names = artifact["top_100_names"]
    ranks = list(range(1, len(top_names) + 1))
    counts = [n["count"] for n in top_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.loglog(ranks, counts, 'o-', color='steelblue', markersize=4)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Zipf Distribution of Names (Top 100)')
    ax1.annotate(f'#1: {top_names[0]["name"]}\n({top_names[0]["count"]:,})', 
                 xy=(1, counts[0]), xytext=(3, counts[0]*0.5),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax2 = axes[1]
    datasets = ['ICE-ID', 'Abt-Buy\n(typical)']
    entropies = [artifact["entropy_bits"], 12.1]
    colors = ['steelblue', 'lightgray']
    
    ax2.bar(datasets, entropies, color=colors, alpha=0.8)
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title(f'Name Ambiguity Comparison\n(ICE-ID: {artifact["effective_names"]:.0f} effective names)')
    ax2.set_ylim(0, 15)
    
    for i, v in enumerate(entropies):
        ax2.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_ambiguity.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig4_ambiguity.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(PLOT_DATA_DIR / "fig4_ambiguity.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    print(f"  Figure 4: Entropy {artifact['entropy_bits']} bits, top-10 share {artifact['top_10_share_pct']}%")


def compute_blocking_curves() -> Dict[str, Any]:
    """Compute real blocking curves on ICE-ID subset."""
    sys.path.insert(0, str(BASE_DIR))
    from bench.data.iceid import IceIdDataset
    from bench.blocking.token_blocking import TokenBlocker
    
    dataset = IceIdDataset(data_dir=str(RAW_DATA.parent))
    split = dataset.load()
    positive_pairs, cluster_labels = dataset.get_ground_truth()
    
    # Sample subset for efficiency
    np.random.seed(42)
    labeled_ids = [rid for rid, cid in cluster_labels.items() if cid != -1]
    sample_persons = list(set(cluster_labels[rid] for rid in labeled_ids))[:2000]
    sample_ids = [rid for rid, cid in cluster_labels.items() if cid in sample_persons]
    
    subset_df = split.records[split.records["id"].isin(sample_ids)].copy()
    subset_split = type(split)(
        name="iceid_subset",
        records=subset_df,
        cluster_labels={k: v for k, v in cluster_labels.items() if k in sample_ids}
    )
    
    # Ground truth pairs in subset
    subset_positive = set()
    person_to_ids = defaultdict(list)
    for rid in sample_ids:
        person_to_ids[cluster_labels[rid]].append(rid)
    for person, ids in person_to_ids.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                subset_positive.add(tuple(sorted((ids[i], ids[j]))))
    
    curves = {}
    
    for blocker_name, fields in [
        ("token_name", ["nafn_norm"]),
        ("token_name_geo", ["nafn_norm", "parish"]),
        ("token_geo_only", ["parish"]),
    ]:
        points = []
        for max_block_size in [50, 100, 200, 500, 1000]:
            blocker = TokenBlocker(fields=fields, min_token_length=2, max_block_size=max_block_size)
            candidates = blocker.block(subset_split, record_ids=sample_ids)
            
            candidate_set = set(tuple(sorted(p)) for p in candidates.pairs)
            recall = len(candidate_set & subset_positive) / len(subset_positive) if subset_positive else 0
            
            points.append({
                "max_block_size": max_block_size,
                "candidates_per_record": round(len(candidates.pairs) / len(sample_ids), 4),
                "blocking_recall": round(recall, 6),
                "n_candidates": len(candidates.pairs),
                "n_ground_truth_pairs": len(subset_positive),
            })
        
        curves[blocker_name] = {
            "description": f"Token blocking on {', '.join(fields)}",
            "fields": fields,
            "points": points,
        }
    
    artifact = {
        "subset_records": len(sample_ids),
        "subset_persons": len(sample_persons),
        "ground_truth_pairs": len(subset_positive),
        "curves": curves,
    }
    
    return artifact


def figure5_blocking(artifact: Dict[str, Any]):
    """Figure 5: Blocking efficiency curves."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = {'token_name': 'steelblue', 'token_name_geo': 'darkorange', 'token_geo_only': 'forestgreen'}
    markers = {'token_name': 'o', 'token_name_geo': '^', 'token_geo_only': 's'}
    
    for name, data in artifact["curves"].items():
        budgets = [p["candidates_per_record"] for p in data["points"]]
        recalls = [p["blocking_recall"] for p in data["points"]]
        ax.semilogx(budgets, recalls, f'{markers[name]}-', color=colors[name], linewidth=2, markersize=6, label=data["description"])
    
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax.text(15, 0.96, '95% recall target', fontsize=9, color='red')
    
    ax.set_xlabel('Candidates per Record')
    ax.set_ylabel('Blocking Recall')
    ax.set_title('ICE-ID: Blocking Efficiency')
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.02)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_blocking.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig5_blocking.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(PLOT_DATA_DIR / "fig5_blocking.json", "w") as f:
        json.dump(artifact, f, indent=2)
    
    print("  Figure 5: Blocking curves generated")


# =============================================================================
# DATASET TABLES (for main_data_paper.tex)
# =============================================================================

def table1_dataset_synopsis():
    """Table 1: Dataset synopsis - one row per dataset."""
    data = [
        {"dataset": "ICE-ID", "time_span": "1703-1920", "n_waves": 16, "n_records": 984028, "n_entities": 226864, "pct_labeled": 50.2, "geo_granularity": "4-level hierarchy", "kinship": "Yes", "license": "CC-BY-4.0", "access": "Open"},
        {"dataset": "Abt-Buy", "time_span": "N/A", "n_waves": 1, "n_records": 11486, "n_entities": 11486, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "Amazon-Google", "time_span": "N/A", "n_waves": 1, "n_records": 13748, "n_entities": 13748, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "DBLP-ACM", "time_span": "N/A", "n_waves": 1, "n_records": 14834, "n_entities": 14834, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "DBLP-Scholar", "time_span": "N/A", "n_waves": 1, "n_records": 34446, "n_entities": 34446, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "Walmart-Amazon", "time_span": "N/A", "n_waves": 1, "n_records": 12288, "n_entities": 12288, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "iTunes-Amazon", "time_span": "N/A", "n_waves": 1, "n_records": 642, "n_entities": 642, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "Beer", "time_span": "N/A", "n_waves": 1, "n_records": 536, "n_entities": 536, "pct_labeled": 100.0, "geo_granularity": "None", "kinship": "No", "license": "Research", "access": "Open"},
        {"dataset": "Fodors-Zagats", "time_span": "N/A", "n_waves": 1, "n_records": 1134, "n_entities": 1134, "pct_labeled": 100.0, "geo_granularity": "Flat (city)", "kinship": "No", "license": "Research", "access": "Open"},
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(TABLE_DATA_DIR / "table1_dataset_synopsis.csv", index=False)
    with open(TABLE_DATA_DIR / "table1_dataset_synopsis.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  Table 1: {len(data)} datasets")


def table2_schema_matrix():
    """Table 2: Schema comparability matrix."""
    feature_families = ["Name / Title", "Age / Birthyear", "Sex / Gender", "Household / Family", "Parent links", "Spouse / Partner", "Address / Geo", "Temporal field", "Free-text notes"]
    datasets = ["ICE-ID", "Abt-Buy", "Amz-Ggl", "DBLP-ACM", "Wlm-Amz", "iTun-Amz", "Beer", "Fod-Zag"]
    
    matrix = {
        "Name / Title": ["Y", "Y", "Y", "Y", "Y", "Y", "Y", "Y"],
        "Age / Birthyear": ["Y", "-", "-", "-", "-", "-", "-", "-"],
        "Sex / Gender": ["Y", "-", "-", "-", "-", "-", "-", "-"],
        "Household / Family": ["Y", "-", "-", "-", "-", "-", "-", "-"],
        "Parent links": ["Y", "-", "-", "-", "-", "-", "-", "-"],
        "Spouse / Partner": ["Y", "-", "-", "-", "-", "-", "-", "-"],
        "Address / Geo": ["Y (4-level)", "-", "-", "-", "-", "-", "-", "Y"],
        "Temporal field": ["Y", "-", "-", "~", "-", "~", "-", "-"],
        "Free-text notes": ["~", "Y", "Y", "-", "Y", "-", "-", "-"],
    }
    
    rows = []
    for feature in feature_families:
        row = {"feature_family": feature}
        for i, ds in enumerate(datasets):
            row[ds] = matrix[feature][i]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DATA_DIR / "table2_schema_matrix.csv", index=False)
    with open(TABLE_DATA_DIR / "table2_schema_matrix.json", "w") as f:
        json.dump({"feature_families": feature_families, "datasets": datasets, "matrix": matrix}, f, indent=2)
    
    print(f"  Table 2: {len(feature_families)} features x {len(datasets)} datasets")


def table3_protocols_splits():
    """Table 3: Evaluation protocols and splits."""
    data = [
        {"dataset": "ICE-ID", "split_rule": "Temporal (pre-1870 / 1871-1890 / 1891-1920)", "train_records": 560334, "train_persons": 153311, "val_records": 147450, "val_persons": 44645, "test_records": 276244, "test_persons": 62109, "positive_def": "Same person ID", "negative_sampling": "2:1 from blocking partition", "transitivity": "Required (cluster labels)", "eval_modes": "Within-wave, Cross-wave"},
        {"dataset": "Abt-Buy", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "Amazon-Google", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "DBLP-ACM", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "DBLP-Scholar", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "Walmart-Amazon", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "iTunes-Amazon", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "Beer", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
        {"dataset": "Fodors-Zagats", "split_rule": "Stratified random", "train_records": "N/A", "train_persons": "N/A", "val_records": "N/A", "val_persons": "N/A", "test_records": "N/A", "test_persons": "N/A", "positive_def": "Match label", "negative_sampling": "Provided", "transitivity": "Not enforced", "eval_modes": "Cross-source"},
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(TABLE_DATA_DIR / "table3_protocols_splits.csv", index=False)
    with open(TABLE_DATA_DIR / "table3_protocols_splits.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  Table 3: {len(data)} protocol definitions")


# =============================================================================
# NARS FIGURES (for main_nars_paper.tex)
# =============================================================================

def nars_rerun_figure():
    """Generate NARS F1 comparison figure from rerun results."""
    results_csv = RESULTS_DIR / "nars_full_eval.csv"
    if not results_csv.exists():
        results_csv = BASE_DIR / "benchmark_results" / "nars_full_eval.csv"
    if not results_csv.exists():
        results_csv = RESULTS_DIR / "nars_rerun_all.csv"
    if not results_csv.exists():
        results_csv = BASE_DIR / "benchmark_results" / "nars_rerun_all.csv"
    
    if not results_csv.exists():
        print("  NARS rerun results not found, skipping figure")
        return
    
    df = pd.read_csv(results_csv)
    
    name_map = {
        'iceid': 'ICE-ID',
        'ice_id': 'ICE-ID',
        'abt_buy': 'Abt-Buy',
        'amazon_google': 'Amazon-Google',
        'dblp_acm': 'DBLP-ACM',
        'dblp_scholar': 'DBLP-Scholar',
        'itunes_amazon': 'iTunes-Amazon',
        'walmart_amazon': 'Walmart-Amazon',
        'beer': 'Beer',
        'fodors_zagats': 'Fodors-Zagats',
    }

    def normalize_name(value: str) -> str:
        key = str(value).strip().lower().replace("-", "_")
        if key == "ice_id":
            return "iceid"
        return key

    df['dataset_key'] = df['dataset'].apply(normalize_name)
    df['dataset_display'] = df['dataset_key'].map(name_map).fillna(df['dataset'])
    df = df.sort_values('f1', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['steelblue' if d != 'ICE-ID' else 'darkorange' for d in df['dataset_display']]
    bars = ax.barh(df['dataset_display'], df['f1'], color=colors, alpha=0.8)
    
    for bar, val in zip(bars, df['f1']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    
    ax.set_xlabel('F1 Score')
    ax.set_title('NARS Performance Across Datasets')
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "nars_rerun_f1.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "nars_rerun_f1.png", bbox_inches='tight', dpi=150)
    plt.close()
    
    with open(PLOT_DATA_DIR / "nars_rerun_f1.json", "w") as f:
        json.dump(df[['dataset', 'f1', 'auc', 'precision', 'recall']].to_dict(orient='records'), f, indent=2)
    
    print("  NARS rerun figure generated")


# =============================================================================
# EXTERNAL DATASET FIGURES
# =============================================================================

def generate_external_dataset_comparison():
    """Generate comparison table and profiles for external datasets."""
    print("\n  Profiling external datasets...")
    
    # Load manifest
    manifest_path = TABLE_DATA_DIR / "external_datasets_manifest.json"
    if not manifest_path.exists():
        print("  External datasets manifest not found. Run fetch_external_datasets.py first.")
        return
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Build comparison table
    from bench.data.external_profiles import profile_external_dataset
    
    comparison = []
    for key, info in manifest.get("datasets", {}).items():
        ds_dir = EXTERNAL_DATA / key
        if not ds_dir.exists():
            continue
        
        profile = profile_external_dataset(key, ds_dir, PLOT_DATA_DIR)
        
        row = {
            "dataset": info.get("name", key),
            "status": info.get("status", "unknown"),
            "type": info.get("type", "unknown"),
            "license": info.get("license", "unknown"),
            "n_files": profile.get("n_files", 0),
            "total_rows": profile.get("total_rows", 0),
            "access": info.get("access", "unknown"),
            "note": info.get("note", ""),
        }
        comparison.append(row)
    
    # Add doc-sourced only
    for key, info in manifest.get("doc_sourced_only", {}).items():
        row = {
            "dataset": info.get("name", key),
            "status": "doc_sourced",
            "type": "restricted",
            "license": info.get("license", "Restricted"),
            "n_files": 0,
            "total_rows": info.get("n_records", "N/A"),
            "access": info.get("access", "Account required"),
            "note": info.get("note", ""),
        }
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df.to_csv(TABLE_DATA_DIR / "table_external_datasets.csv", index=False)
    with open(TABLE_DATA_DIR / "table_external_datasets.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"  External dataset comparison: {len(comparison)} datasets")


def generate_missingno_plots():
    """Generate missingno-style missingness plots for ICE-ID and external datasets."""
    print("\n  Generating missingness matrix plots...")
    
    try:
        import missingno as msno
    except ImportError:
        print("  missingno not installed, skipping missingness plots")
        return
    
    # ICE-ID missingness
    people_csv = RAW_DATA / "people.csv"
    if people_csv.exists():
        sample = pd.read_csv(people_csv, nrows=1000, low_memory=False)
        fig, ax = plt.subplots(figsize=(12, 8))
        msno.matrix(sample, ax=ax, sparkline=False, fontsize=10)
        plt.title('ICE-ID: Missing Data Pattern (sample)')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missingno_iceid.pdf", bbox_inches='tight')
        plt.savefig(FIGURES_DIR / "missingno_iceid.png", bbox_inches='tight', dpi=150)
        plt.close()
        print("    ICE-ID missingness plot generated")
    
    # External datasets
    for ds_key in ["synthea", "febrl"]:
        ds_dir = EXTERNAL_DATA / ds_key / "raw"
        if not ds_dir.exists():
            continue
        
        csv_files = list(ds_dir.glob("*.csv"))
        if not csv_files:
            continue
        
        sample = pd.read_csv(csv_files[0], nrows=500, low_memory=False)
        if len(sample.columns) > 3:
            fig, ax = plt.subplots(figsize=(12, 8))
            msno.matrix(sample, ax=ax, sparkline=False, fontsize=10)
            plt.title(f'{ds_key.upper()}: Missing Data Pattern (sample)')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"missingno_{ds_key}.pdf", bbox_inches='tight')
            plt.savefig(FIGURES_DIR / f"missingno_{ds_key}.png", bbox_inches='tight', dpi=150)
            plt.close()
            print(f"    {ds_key} missingness plot generated")


def table_longitudinal_comparison():
    """Table: Longitudinal dataset comparison (temporal signal, spans, entities)."""
    data = [
        {"dataset": "ICE-ID", "time_span": "1703-1920", "temporal_signal": "Census year", "n_entities": "227K persons", "entity_type": "Person", "temporal_coverage": "16 waves", "file_derived": "Yes", "access": "Open"},
        {"dataset": "IPUMS LRS", "time_span": "1850-1940", "temporal_signal": "Census year", "n_entities": "~50M", "entity_type": "Person", "temporal_coverage": "10 waves", "file_derived": "No (doc)", "access": "Account"},
        {"dataset": "IPUMS MLP", "time_span": "1870-2020", "temporal_signal": "Census + survey", "n_entities": "~100M", "entity_type": "Person", "temporal_coverage": "Continuous", "file_derived": "No (doc)", "access": "Account"},
        {"dataset": "IPUMS NAPP", "time_span": "1801-1910", "temporal_signal": "Census year", "n_entities": "~100M", "entity_type": "Person", "temporal_coverage": "Multi-country", "file_derived": "No (doc)", "access": "Account"},
        {"dataset": "ORCID", "time_span": "2012-present", "temporal_signal": "Last modified", "n_entities": "~18M", "entity_type": "Researcher", "temporal_coverage": "Continuous", "file_derived": "Yes (sample)", "access": "Open"},
        {"dataset": "SemParl", "time_span": "1907-2021", "temporal_signal": "Speech date", "n_entities": "~7K persons", "entity_type": "Parliamentarian", "temporal_coverage": "Daily", "file_derived": "Yes", "access": "Open"},
        {"dataset": "CKCC", "time_span": "1600-1800", "temporal_signal": "Letter date", "n_entities": "~5K persons", "entity_type": "Correspondent", "temporal_coverage": "Sparse", "file_derived": "Yes", "access": "Open"},
        {"dataset": "correspSearch", "time_span": "1500-2000", "temporal_signal": "Letter date", "n_entities": "~130K letters", "entity_type": "Correspondent", "temporal_coverage": "Sparse", "file_derived": "Yes", "access": "Open"},
        {"dataset": "Synthea", "time_span": "1950-2020", "temporal_signal": "Encounter date", "n_entities": "~1K (sample)", "entity_type": "Patient", "temporal_coverage": "Daily", "file_derived": "Yes", "access": "Open"},
        {"dataset": "FEBRL", "time_span": "N/A", "temporal_signal": "None", "n_entities": "5K-10K", "entity_type": "Patient", "temporal_coverage": "Static", "file_derived": "Yes", "access": "Open"},
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(TABLE_DATA_DIR / "table_longitudinal_comparison.csv", index=False)
    with open(TABLE_DATA_DIR / "table_longitudinal_comparison.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  Table: Longitudinal comparison: {len(data)} datasets")


def generate_external_artifacts():
    """Generate all external dataset artifacts."""
    generate_external_dataset_comparison()
    generate_missingno_plots()
    table_longitudinal_comparison()


# =============================================================================
# MAIN
# =============================================================================

def generate_dataset_figures():
    print("\n1. Computing temporal coverage...")
    temporal = compute_temporal_coverage()
    figure1_temporal_coverage(temporal)
    
    print("\n2. Computing missingness...")
    missingness = compute_missingness()
    figure2_missingness(missingness)
    
    print("\n3. Computing cluster sizes...")
    clusters = compute_cluster_sizes()
    figure3_cluster_sizes(clusters)
    
    print("\n4. Computing name ambiguity...")
    ambiguity = compute_name_ambiguity()
    figure4_ambiguity(ambiguity)
    
    print("\n5. Computing blocking curves...")
    blocking = compute_blocking_curves()
    figure5_blocking(blocking)


def generate_dataset_tables():
    print("\n1. Table 1: Dataset synopsis...")
    table1_dataset_synopsis()
    
    print("\n2. Table 2: Schema matrix...")
    table2_schema_matrix()
    
    print("\n3. Table 3: Protocols and splits...")
    table3_protocols_splits()


def generate_nars_artifacts():
    print("\n1. NARS rerun figure...")
    nars_rerun_figure()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("target", nargs="?", default="all", choices=["all", "figures", "tables", "nars", "external"],
                        help="What to generate: all, figures, tables, nars, or external")
    args = parser.parse_args()
    
    ensure_dirs()
    
    print("=" * 70)
    print("GENERATING PAPER ARTIFACTS")
    print("=" * 70)
    
    if args.target in ["all", "figures"]:
        print("\n>>> DATASET FIGURES")
        generate_dataset_figures()
    
    if args.target in ["all", "tables"]:
        print("\n>>> DATASET TABLES")
        generate_dataset_tables()
    
    if args.target in ["all", "nars"]:
        print("\n>>> NARS FIGURES")
        generate_nars_artifacts()
    
    if args.target in ["all", "external"]:
        print("\n>>> EXTERNAL DATASET ARTIFACTS")
        generate_external_artifacts()
    
    print("\n" + "=" * 70)
    print("DONE")
    print(f"Figures: {FIGURES_DIR}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
