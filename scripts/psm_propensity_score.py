"""Generate propensity scores for M6 Propensity Score Matching (PSM).

Run from project root after M4 provides the treatment flag file:
    python scripts/psm_propensity_score.py --config config/paths.yaml

Expected M4 input:
    models/psm_inputs/psm_treatment_flags.csv

Required columns:
    household_key,is_treated,treatment_source,treatment_cutoff_day

Output for M6:
    models/m6_handoff/propensity_scores_for_psm.csv

Method:
- Logistic Regression estimates P(is_treated = 1 | pre-outcome covariates).
- The script intentionally excludes churn/outcome/future/profit/coupon-like columns
  from the propensity model to reduce leakage and collider bias risk.
- This is a PSM support file, not a replacement for the core M5 churn model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.utils import ensure_project_structure, find_project_root, load_config, resolve_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate propensity scores for M6 PSM.")
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument(
        "--allow-dummy",
        action="store_true",
        help="Allow running on a DUMMY_PLACEHOLDER_DO_NOT_REPORT treatment file for dry-run testing only.",
    )
    parser.add_argument(
        "--with-cv",
        action="store_true",
        help="Optionally estimate CV ROC-AUC/PR-AUC for the propensity model. Slower; not required for M6 handoff.",
    )
    return parser.parse_args()


def _safe_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _resolve_output_path(raw_path: str | Path, project_root: Path) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else project_root / p


def load_treatment_flags(path: Path, id_col: str, treatment_col: str, allow_dummy: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"PSM treatment flag file not found: {path}\n"
            "Ask M4 to provide models/psm_inputs/psm_treatment_flags.csv with columns: "
            f"{id_col},{treatment_col},treatment_source,treatment_cutoff_day.\n"
            "For dry-run only, run: python scripts/create_dummy_psm_treatment_flags.py --overwrite"
        )

    flags = pd.read_csv(path)
    required = {id_col, treatment_col}
    missing = required - set(flags.columns)
    if missing:
        raise ValueError(f"Treatment flag file missing required columns: {sorted(missing)}")

    flags[id_col] = flags[id_col].astype(int)
    if flags[id_col].duplicated().any():
        dup = int(flags[id_col].duplicated().sum())
        raise ValueError(f"Treatment flag file has duplicated {id_col}: {dup} duplicate rows")

    flags[treatment_col] = flags[treatment_col].astype(int)
    unique_values = set(flags[treatment_col].dropna().unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(f"{treatment_col} must be binary 0/1, found: {sorted(unique_values)}")
    if len(unique_values) < 2:
        raise ValueError(f"{treatment_col} has only one class. PSM needs both treated and control households.")

    if "treatment_source" in flags.columns:
        is_dummy = flags["treatment_source"].astype(str).str.contains("DUMMY_PLACEHOLDER_DO_NOT_REPORT", case=False, na=False).any()
        if is_dummy and not allow_dummy:
            raise ValueError(
                "The treatment flag file is marked as DUMMY_PLACEHOLDER_DO_NOT_REPORT. "
                "Use --allow-dummy only for code dry-run. Do not report dummy PSM scores."
            )

    return flags


def should_exclude_column(col: str, id_col: str, target_col: str, treatment_col: str, keywords: Iterable[str]) -> bool:
    c = col.lower()
    if col in {id_col, target_col, treatment_col}:
        return True
    return any(k.lower() in c for k in keywords)


def select_covariates(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    treatment_col: str,
    exclude_keywords: Iterable[str],
) -> List[str]:
    covariates = [
        c
        for c in df.columns
        if not should_exclude_column(c, id_col, target_col, treatment_col, exclude_keywords)
    ]
    if not covariates:
        raise ValueError("No covariates left after exclusion rules. Review psm.exclude_feature_keywords in config.")
    return covariates


def build_propensity_pipeline(df: pd.DataFrame, covariates: List[str]) -> Pipeline:
    cat_cols = [c for c in covariates if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    num_cols = [c for c in covariates if c not in cat_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", _safe_one_hot_encoder()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def estimate_cv_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int) -> Dict[str, float | None]:
    min_class_count = int(y.value_counts().min())
    n_splits = min(cv_folds, min_class_count)
    if n_splits < 2:
        return {"cv_roc_auc": None, "cv_pr_auc": None, "cv_folds_used": None}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        p_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        return {
            "cv_roc_auc": float(roc_auc_score(y, p_cv)),
            "cv_pr_auc": float(average_precision_score(y, p_cv)),
            "cv_folds_used": int(n_splits),
        }
    except Exception as exc:
        print(f"[PSM] CV metric estimation skipped: {exc}")
        return {"cv_roc_auc": None, "cv_pr_auc": None, "cv_folds_used": int(n_splits)}


def add_optional_m5_fields(out: pd.DataFrame, paths: Dict[str, Path], id_col: str) -> pd.DataFrame:
    """Merge useful M5 risk fields if priority_customers_all.csv exists."""
    priority_path = paths.get("models_m6_handoff_dir", paths["models_dir"] / "m6_handoff") / "priority_customers_all.csv"
    if not priority_path.exists():
        return out
    try:
        cols = [
            id_col,
            "p_churn_calibrated",
            "risk_rank",
            "risk_decile",
            "priority_segment",
            "predicted_discounted_value_60d_if_active",
        ]
        available = pd.read_csv(priority_path, nrows=1).columns.tolist()
        usecols = [c for c in cols if c in available]
        if id_col not in usecols:
            return out
        priority = pd.read_csv(priority_path, usecols=usecols)
        priority[id_col] = priority[id_col].astype(int)
        return out.merge(priority, on=id_col, how="left")
    except Exception as exc:
        print(f"[PSM] Optional M5 risk-field merge skipped: {exc}")
        return out


def main() -> None:
    args = parse_args()
    
    try:
        project_root = find_project_root()
    except FileNotFoundError:
        # Some GitHub-ready zips do not include the Data/ folder.
        # In that case, this script still runs from the repository root inferred from its own location.
        project_root = PROJECT_ROOT_FOR_IMPORT

    config = load_config(config_path=args.config, project_root=project_root)
    paths = resolve_project_paths(config, project_root)
    ensure_project_structure(paths)

    modeling_cfg = config.get("modeling", {})
    psm_cfg = config.get("psm", {})
    id_col = str(modeling_cfg.get("id_col", "household_key"))
    target_col = str(modeling_cfg.get("target_col", "churn_flag"))
    treatment_col = str(psm_cfg.get("treatment_col", "is_treated"))
    random_state = int(psm_cfg.get("random_state", modeling_cfg.get("random_state", 42)))
    cv_folds = int(psm_cfg.get("cv_folds", 5))
    exclude_keywords = list(psm_cfg.get("exclude_feature_keywords", []))

    feature_path = paths.get(
        "psm_feature_table_csv",
        paths.get("feature_table_linear_csv", paths["feature_table_csv"]),
    )
    treatment_path = paths.get("psm_treatment_flags_csv", project_root / "models" / "psm_inputs" / "psm_treatment_flags.csv")
    output_scores_path = _resolve_output_path(
        psm_cfg.get("output_propensity_scores_csv", "models/m6_handoff/propensity_scores_for_psm.csv"),
        project_root,
    )
    output_summary_path = _resolve_output_path(
        psm_cfg.get("output_model_summary_csv", "models/diagnostics/psm_propensity_model_summary.csv"),
        project_root,
    )
    output_covariates_path = _resolve_output_path(
        psm_cfg.get("output_covariates_used_csv", "models/diagnostics/psm_covariates_used.csv"),
        project_root,
    )
    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_covariates_path.parent.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(feature_path)
    features[id_col] = features[id_col].astype(int)
    features = features.drop_duplicates(id_col).reset_index(drop=True)

    flags = load_treatment_flags(treatment_path, id_col, treatment_col, allow_dummy=args.allow_dummy)
    df = features.merge(flags, on=id_col, how="inner")
    if len(df) != len(features):
        missing_count = len(features) - len(df)
        raise ValueError(
            f"Treatment flags do not cover all feature rows. Missing rows after merge: {missing_count}. "
            "Ask M4 to include every household_key from the configured PSM feature table."
        )

    covariates = select_covariates(df, id_col, target_col, treatment_col, exclude_keywords)
    X = df[covariates].copy()
    y = df[treatment_col].astype(int)

    model = build_propensity_pipeline(df, covariates)
    estimate_cv = bool(psm_cfg.get("estimate_cv_metrics", False)) or bool(args.with_cv)
    if estimate_cv:
        cv_metrics = estimate_cv_metrics(model, X, y, cv_folds=cv_folds, random_state=random_state)
    else:
        cv_metrics = {"cv_roc_auc": None, "cv_pr_auc": None, "cv_folds_used": None}
    model.fit(X, y)
    propensity_score = model.predict_proba(X)[:, 1]
    propensity_score = np.clip(propensity_score, 1e-6, 1 - 1e-6)
    propensity_logit = np.log(propensity_score / (1 - propensity_score))

    treated_scores = propensity_score[y == 1]
    control_scores = propensity_score[y == 0]
    common_low = float(max(treated_scores.min(), control_scores.min()))
    common_high = float(min(treated_scores.max(), control_scores.max()))

    out = pd.DataFrame(
        {
            id_col: df[id_col].astype(int),
            treatment_col: y,
            "propensity_score": propensity_score,
            "propensity_logit": propensity_logit,
            "common_support_flag": (propensity_score >= common_low) & (propensity_score <= common_high),
        }
    )
    for extra_col in ["treatment_source", "treatment_cutoff_day"]:
        if extra_col in df.columns:
            out[extra_col] = df[extra_col]
    out = add_optional_m5_fields(out, paths, id_col)
    out.to_csv(output_scores_path, index=False)

    covariate_df = pd.DataFrame(
        {
            "covariate": covariates,
            "dtype": [str(df[c].dtype) for c in covariates],
        }
    )
    covariate_df.to_csv(output_covariates_path, index=False)

    summary = {
        "status": "completed",
        "method": "LogisticRegression propensity model",
        "feature_table": str(feature_path.relative_to(project_root)),
        "treatment_flag_file": str(treatment_path.relative_to(project_root)),
        "output_file_for_m6": str(output_scores_path.relative_to(project_root)),
        "n_households": int(len(df)),
        "treated_count": int(y.sum()),
        "control_count": int((y == 0).sum()),
        "treated_share": float(y.mean()),
        "covariate_count": int(len(covariates)),
        "excluded_keyword_rules": exclude_keywords,
        "propensity_score_min": float(propensity_score.min()),
        "propensity_score_max": float(propensity_score.max()),
        "propensity_score_mean": float(propensity_score.mean()),
        "common_support_low": common_low,
        "common_support_high": common_high,
        "common_support_share": float(out["common_support_flag"].mean()),
        "cv_roc_auc": cv_metrics["cv_roc_auc"],
        "cv_pr_auc": cv_metrics["cv_pr_auc"],
        "cv_folds_used": cv_metrics["cv_folds_used"],
        "important_note": (
            "PSM supports quasi-experimental analysis on observed covariates only. "
            "It does not prove randomized causal effect, especially if treatment means coupon redemption/usage rather than random coupon assignment."
        ),
    }
    pd.DataFrame([summary]).to_csv(output_summary_path, index=False)

    note_path = paths["reports_internal_dir"] / "M5_PSM_propensity_score_note.md"
    note_path.write_text(
        "# M5 PSM Propensity Score Handoff\n\n"
        "M5 estimates `propensity_score = P(is_treated = 1 | observed pre-outcome covariates)` "
        "for M6's Propensity Score Matching workflow.\n\n"
        f"- Input treatment flag file: `{treatment_path.relative_to(project_root)}`\n"
        f"- Output for M6: `{output_scores_path.relative_to(project_root)}`\n"
        f"- Covariates used: `{output_covariates_path.relative_to(project_root)}`\n"
        f"- Model summary: `{output_summary_path.relative_to(project_root)}`\n\n"
        "## M4 dependency\n\n"
        "Replace the dummy/placeholder treatment file with M4's real treatment flag file before final PSM reporting. "
        "Required columns: `household_key`, `is_treated`, `treatment_source`, `treatment_cutoff_day`.\n\n"
        "## Interpretation warning\n\n"
        "This is not a randomized A/B test. PSM can balance observed covariates, but unobserved confounding may remain. "
        "Write the conclusion as association after matching, not definitive causal proof.\n",
        encoding="utf-8",
    )

    print("[PSM] Propensity score generation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
