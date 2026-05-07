"""
Organize M5-generated outputs into purpose-based folders under models/.

Important scope rule:
- This script only organizes M5-owned outputs.
- It does NOT move M4 handoff / feature artifacts such as:
  - final_ML_features.csv
  - final_features.parquet
  - final_train_features.parquet
  - final_test_features.parquet

Recommended usage from project root:
    python scripts/organize_m5_outputs.py
"""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = MODELS_DIR / "reports"
M6_HANDOFF_DIR = MODELS_DIR / "m6_handoff"
DIAGNOSTICS_DIR = MODELS_DIR / "diagnostics"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"


# ---------------------------------------------------------------------
# Files owned by M4 / upstream feature engineering.
# Do not move these in M5 output organization.
# ---------------------------------------------------------------------
PROTECTED_UPSTREAM_FILES = {
    "final_ML_features.csv",
    "final_features.parquet",
    "final_train_features.parquet",
    "final_test_features.parquet",
}


# ---------------------------------------------------------------------
# M5 outputs used mainly for report / reviewer understanding.
# These are lightweight summary files, not customer-level handoff files.
# ---------------------------------------------------------------------
REPORT_OUTPUTS = {
    "model_metrics.csv",
    "champion_test_metrics.csv",
    "calibration_summary.csv",
    "top_k_precision_summary.csv",
    "ranking_decile_performance.csv",
    "scenario_profit_summary.csv",
    "profit_threshold_analysis.csv",
    "value_model_metrics.csv",
    "shap_global_importance.csv",
    "feature_importance.csv",
}


M6_HANDOFF_OUTPUTS = {
    "churn_predictions.csv",
    "priority_customers_all.csv",
    "high_risk_customers_for_ab_test.csv",
}


# ---------------------------------------------------------------------
# M5 diagnostics / internal audit files.
# Useful for debugging, methodology checks, and defending decisions.
# ---------------------------------------------------------------------
DIAGNOSTIC_OUTPUTS = {
    "model_tuning_results.csv",
    "model_cv_summary.csv",
    "calibration_by_decile.csv",
    "calibration_flatspot_diagnostics.csv",
    "active_model_metrics.csv",
    "active_churn_target_overlap_audit.csv",
    "discounted_value_labels.csv",
    "scenario_sensitivity_grid.csv",
    "break_even_analysis.csv",
    "shap_top_risk_customer_reasons.csv",
    "seasonality_audit.csv",
    "seasonality_window_comparison.csv",
    "recommendation_merge_audit.csv",
    "voucher_diversification_audit.csv",
    "multicollinearity_vif.csv",
    "numeric_feature_correlation_pairs.csv",
    "value_model_residual_summary.csv",
    "value_model_decile_diagnostics.csv",
    "split_stability_runs.csv",
    "split_stability_summary.csv",
    "feature_lineage_audit_template.csv",
}


# ---------------------------------------------------------------------
# M5 model artifacts.
# Do not include M4 final feature files here.
# ---------------------------------------------------------------------
ARTIFACT_OUTPUTS = {
    "model.pkl",
}


def ensure_dirs() -> None:
    """Create target folders if they do not exist."""
    for folder in [REPORTS_DIR, M6_HANDOFF_DIR, DIAGNOSTICS_DIR, ARTIFACTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def move_if_exists(filename: str, target_dir: Path) -> bool:
    """
    Move a file from models root to target_dir if it exists.

    Returns:
        True if moved, False if not found.
    """
    source = MODELS_DIR / filename
    target = target_dir / filename

    if not source.exists():
        return False

    if filename in PROTECTED_UPSTREAM_FILES:
        print(f"[SKIP: upstream/M4-owned] {filename}")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)

    if target.exists():
        target.unlink()

    shutil.move(str(source), str(target))
    print(f"[MOVED] {filename} -> {target.relative_to(PROJECT_ROOT)}")
    return True


def organize_outputs() -> None:
    """Organize M5 outputs into subfolders."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models folder not found: {MODELS_DIR}")

    ensure_dirs()

    moved_count = 0

    for filename in sorted(REPORT_OUTPUTS):
        moved_count += int(move_if_exists(filename, REPORTS_DIR))

    for filename in sorted(M6_HANDOFF_OUTPUTS):
        moved_count += int(move_if_exists(filename, M6_HANDOFF_DIR))

    for filename in sorted(DIAGNOSTIC_OUTPUTS):
        moved_count += int(move_if_exists(filename, DIAGNOSTICS_DIR))

    for filename in sorted(ARTIFACT_OUTPUTS):
        moved_count += int(move_if_exists(filename, ARTIFACTS_DIR))

    # Explicitly log protected files that remain in models root if present.
    for filename in sorted(PROTECTED_UPSTREAM_FILES):
        if (MODELS_DIR / filename).exists():
            print(f"[KEPT IN ROOT: upstream/M4-owned] {filename}")

    print(f"\nDone. Moved {moved_count} M5-owned output file(s).")


if __name__ == "__main__":
    organize_outputs()
