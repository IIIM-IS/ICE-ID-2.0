#!/usr/bin/env python3
"""
Consolidated data preparation script.

Usage:
    python scripts/prepare_data.py deepmatcher    # Download DeepMatcher datasets
    python scripts/prepare_data.py all            # Prepare all data
"""
import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

DEEPMATCHER_DATA = BASE_DIR / "deepmatcher_data"


def prepare_deepmatcher():
    """Download all DeepMatcher datasets."""
    from bench.data.deepmatcher import DeepMatcherDataset
    
    datasets = ["abt_buy", "amazon_google", "dblp_acm", "dblp_scholar", 
                "itunes_amazon", "walmart_amazon", "beer", "fodors_zagats"]
    
    print("Downloading DeepMatcher datasets...")
    for ds_name in datasets:
        print(f"\n  {ds_name}...")
        try:
            dm = DeepMatcherDataset(ds_name, data_dir=str(DEEPMATCHER_DATA))
            dm.download()
            print(f"    OK")
        except Exception as e:
            print(f"    Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark data")
    parser.add_argument("target", nargs="?", default="all",
                        choices=["all", "deepmatcher"],
                        help="What data to prepare")
    args = parser.parse_args()
    
    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)
    
    if args.target in ["all", "deepmatcher"]:
        prepare_deepmatcher()
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

