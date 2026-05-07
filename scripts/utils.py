"""Utility helpers for project paths and configuration.

These helpers keep notebooks and scripts independent from the current working
folder. They are intentionally lightweight so the repo remains easy for all
team members to run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "DDM-Churn-Project"},
    "paths": {
        "data_raw_dir": "Data/Raw",
        "data_processed_dir": "Data/Processed",
        "intermediate_features_dir": "Data/Intermediate/features",
        "intermediate_analysis_dir": "Data/Intermediate/analysis",
        "models_dir": "models",
        "notebooks_dir": "notebooks",
        "scripts_dir": "scripts",
        "visualization_exports_dir": "visualization/exports",
        "reports_internal_dir": "reports/internal_briefs",
        "reports_final_paper_dir": "reports/final_paper",
    },
    "inputs": {
        "feature_table_csv": "models/final_ML_features.csv",
        "transaction_master_parquet": "Data/Processed/transactions_master.parquet",
        "customer_base_parquet": "Data/Processed/customer_base_labeled.parquet",
        "voucher_recommendations_csv": "Data/Intermediate/market_basket/personalized_voucher_recommendations.csv",
    },
    "modeling": {
        "id_col": "household_key",
        "target_col": "churn_flag",
        "categorical_cols": ["Primary_Store_ID"],
        "cut_off_day": 651,
        "test_size": 0.20,
        "validation_size": 0.25,
        "random_state": 42,
        "f_beta": 2,
        "n_estimators": 150,
        "n_jobs": 1,
    },
    "expected_profit_scenarios": {
        "conservative": {"gross_margin": 0.20, "retention_lift": 0.03, "treatment_cost": 5.0},
        "base": {"gross_margin": 0.25, "retention_lift": 0.05, "treatment_cost": 5.0},
        "optimistic": {"gross_margin": 0.30, "retention_lift": 0.08, "treatment_cost": 5.0},
    },
}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary without mutating the original input."""
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root by walking up until Data/ and notebooks/ exist."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "Data").exists() and (candidate / "notebooks").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing both Data/ and notebooks/.")


def load_config(config_path: str | Path | None = None, project_root: Path | None = None) -> Dict[str, Any]:
    """Load YAML config if available; otherwise return defaults."""
    root = project_root or find_project_root()
    cfg = DEFAULT_CONFIG
    if config_path is None:
        config_path = root / "config" / "paths.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if config_path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load config/paths.yaml. Install with `pip install pyyaml`.") from exc
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_update(DEFAULT_CONFIG, loaded)
    return cfg


def resolve_project_paths(config: Dict[str, Any], project_root: Path) -> Dict[str, Path]:
    """Resolve all path strings in config to absolute Path objects."""
    paths: Dict[str, Path] = {"project_root": project_root}
    for section in ["paths", "inputs"]:
        for key, value in config.get(section, {}).items():
            p = Path(value)
            paths[key] = p if p.is_absolute() else project_root / p
    return paths


def ensure_project_structure(paths: Dict[str, Path]) -> None:
    """Create standard output directories used by the project.

    Any resolved config key ending with ``_dir`` is treated as an output
    directory. This lets M5 define subfolders such as ``models/reports`` and
    ``models/m6_handoff`` without requiring a hard-coded update here.
    """
    for key, path in paths.items():
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)
