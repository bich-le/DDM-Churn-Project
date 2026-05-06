\
"""Lightweight data-processing utilities for repo validation.

This script does not replace M1/M4 notebooks. It provides a simple CLI sanity
check so team members can verify that the standard project folders and core M5
inputs exist before running predictive modeling.

Run:
    python scripts/data_processing.py --config config/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.utils import ensure_project_structure, find_project_root, load_config, resolve_project_paths


def validate_core_inputs(config_path: str | Path | None = None) -> Dict[str, Any]:
    project_root = find_project_root()
    config = load_config(config_path=config_path, project_root=project_root)
    paths = resolve_project_paths(config, project_root)
    ensure_project_structure(paths)

    checks = {}
    for key in ["feature_table_csv", "transaction_master_parquet", "customer_base_parquet"]:
        path = paths[key]
        checks[key] = {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}

    out_path = paths["reports_internal_dir"] / "repo_input_validation.json"
    out_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DDM Churn Project core input files and folders.")
    parser.add_argument("--config", default="config/paths.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_core_inputs(args.config)
