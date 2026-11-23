"""Resolve the canonical research release consistently across tabs."""
import json
import os
from pathlib import Path


def selected_release_root(iceid_root):
    explicit=os.environ.get('ICEID_RELEASE_DIR')
    if explicit:return Path(explicit).resolve()
    config=Path(iceid_root).resolve().parent/'label-repair/pipeline_paths.json'
    if config.is_file():
        selected=json.loads(config.read_text()).get('selected_release')
        if not selected:raise ValueError('Canonical pipeline configuration has no selected release')
        return (config.parent/selected).resolve()
    return None


def default_data_paths(iceid_root):
    root=selected_release_root(iceid_root)
    if root is None:
        raw = Path(iceid_root) / "raw_data"
        return str(raw / "people.csv"), str(raw / "manntol_einstaklingar_new.csv")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Selected reference release is incomplete")
    paths = json.loads((root / "application_paths.json").read_text())
    result = tuple(str((root / paths[key]).resolve()) for key in ["people_csv", "labels_csv"])
    if not all(Path(path).is_file() for path in result):
        raise FileNotFoundError("Selected release data/label mapping is missing")
    return result


def default_run_directory(iceid_root):
    root=selected_release_root(iceid_root)
    if root is not None:
        paths=json.loads((root/'application_paths.json').read_text())
        return str(root/paths.get('model_outputs_dir','application_runs'))
    return str(Path(iceid_root)/'runs/hundred_loose_dual/shard_s0_of_1')
