"""Command-line interface for er-bench."""

import argparse
import sys
import os

import yaml

from .runner import run_experiment, run_grid
from .config.schema import load_config, validate_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ER-Bench: Entity Resolution Benchmark Framework"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("config", help="Path to config YAML file")
    run_parser.add_argument("-o", "--output", default="results", help="Output directory")
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    grid_parser = subparsers.add_parser("grid", help="Run grid of experiments")
    grid_parser.add_argument("config", help="Path to grid config YAML file")
    grid_parser.add_argument("-o", "--output", default="results", help="Output directory")
    grid_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument("component", choices=["datasets", "models", "blockers", "calibrators", "clusterers"])
    
    args = parser.parse_args()
    
    if args.command == "run":
        config = load_config(args.config)
        errors = validate_config(config)
        if errors:
            print("Configuration errors:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        
        result = run_experiment(config, args.output, verbose=args.verbose)
        print(f"\nResults saved to: {args.output}")
        
    elif args.command == "grid":
        config = load_config(args.config)
        results = run_grid(config, args.output, verbose=args.verbose)
        print(f"\nGrid results saved to: {args.output}/grid_results.csv")
        
    elif args.command == "list":
        from .core.registry import get_registry
        registry = get_registry(args.component)
        print(f"Available {args.component}:")
        for name in registry.list():
            print(f"  - {name}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

