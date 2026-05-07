"""M5 v3 modeling pipeline: calibrated churn, two-part discounted value, SHAP, and seasonality audit.

Run from the project root:
    python scripts/modeling.py --config config/paths.yaml

Scope:
- M5-only changes. This script does not rebuild or modify M4 feature engineering.
- Uses M4's delivered feature table as input.
- Adds discounted 60-day value labels from post-cutoff transactions.
- Uses a two-part value model: P(future active) × E(discounted value | active).
- Uses calibrated churn probability for business decisioning.
- Adds SHAP explainability and seasonality audit outputs when dependencies allow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.evaluation import (
    best_fbeta_threshold,
    calibration_by_decile,
    evaluate_proba,
    extract_feature_importance,
    make_one_hot_encoder,
    profit_threshold_analysis,
    ranking_decile_performance,
    regression_metrics,
    top_k_precision_summary,
)
from scripts.utils import (
    ensure_project_structure,
    find_project_root,
    load_config,
    resolve_project_paths,
)

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(exc)

try:
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline

    IMBLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    SMOTETomek = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False
    IMBLEARN_IMPORT_ERROR = str(exc)

XGBClassifier = None
XGBRegressor = None


class PrefitProbabilityCalibrator:
    """Calibrate probabilities from an already-fitted probabilistic model."""

    def __init__(self, base_model: Any, method: str = "sigmoid") -> None:
        self.base_model = base_model
        self.method = method
        self.calibrator: Any | None = None

    def fit(
        self, X_calib: pd.DataFrame, y_calib: pd.Series
    ) -> "PrefitProbabilityCalibrator":
        raw = self.base_model.predict_proba(X_calib)[:, 1]
        y = np.asarray(y_calib).astype(int)
        if self.method == "sigmoid":
            self.calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
            self.calibrator.fit(raw.reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        if self.calibrator is None:
            cal = raw
        elif self.method == "sigmoid":
            cal = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            cal = self.calibrator.predict(raw)
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])


def load_feature_table(
    feature_path: Path, id_col: str, target_col: str, categorical_cols: List[str]
) -> pd.DataFrame:
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")
    features = pd.read_csv(feature_path)
    required = {id_col, target_col}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {sorted(missing)}")
    features[id_col] = features[id_col].astype(int)
    features[target_col] = features[target_col].astype(int)
    for col in categorical_cols:
        if col in features.columns:
            features[col] = features[col].astype(str)
    return features.drop_duplicates(id_col).reset_index(drop=True)


def audit_inputs(
    features: pd.DataFrame,
    paths: Dict[str, Path],
    id_col: str,
    target_col: str,
    cut_off_day: int,
) -> pd.DataFrame:
    rows: List[Tuple[str, Any]] = [
        ("feature_rows", len(features)),
        ("feature_columns", features.shape[1]),
        ("duplicate_household_key", int(features[id_col].duplicated().sum())),
        ("missing_values_total", int(features.isna().sum().sum())),
        ("churn_count", int(features[target_col].sum())),
        ("non_churn_count", int((features[target_col] == 0).sum())),
        ("churn_rate", float(features[target_col].mean())),
        ("cut_off_day_used_by_m5", int(cut_off_day)),
    ]
    customer_path = paths["customer_base_parquet"]
    txn_path = paths["transaction_master_parquet"]
    if customer_path.exists():
        customers = pd.read_parquet(customer_path)
        customers[id_col] = customers[id_col].astype(int)
        missing = sorted(set(customers[id_col]) - set(features[id_col]))
        rows += [
            ("customer_base_rows", len(customers)),
            ("missing_from_features_count", len(missing)),
            ("missing_from_features_households", ", ".join(map(str, missing[:50]))),
        ]
        if missing and txn_path.exists():
            txns = pd.read_parquet(txn_path, columns=[id_col, "DAY", "SALES_VALUE"])
            txns[id_col] = txns[id_col].astype(int)
            miss_txns = txns[txns[id_col].isin(missing)]
            if not miss_txns.empty:
                missing_df = (
                    miss_txns.groupby(id_col)
                    .agg(
                        txn_rows=("DAY", "size"),
                        first_day=("DAY", "min"),
                        last_day=("DAY", "max"),
                        pre_cutoff_txn_rows=(
                            "DAY",
                            lambda s: int((s < cut_off_day).sum()),
                        ),
                        post_cutoff_txn_rows=(
                            "DAY",
                            lambda s: int((s >= cut_off_day).sum()),
                        ),
                        total_sales=("SALES_VALUE", "sum"),
                    )
                    .reset_index()
                )
                missing_df.to_csv(
                    paths["intermediate_analysis_dir"]
                    / "m5_missing_household_investigation.csv",
                    index=False,
                )
    audit = pd.DataFrame(rows, columns=["check", "value"])
    audit.to_csv(paths["reports_internal_dir"] / "m5_data_audit.csv", index=False)
    return audit


def prepare_splits(
    features: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_cols: List[str],
    test_size: float,
    validation_size: float,
    random_state: int,
) -> Dict[str, Any]:
    X = features[feature_cols].copy()
    y = features[target_col].astype(int).copy()
    ids = features[id_col].astype(int).copy()
    X_trainval, X_test, y_trainval, y_test, ids_trainval, ids_test = train_test_split(
        X, y, ids, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_trainval,
        y_trainval,
        ids_trainval,
        test_size=validation_size,
        stratify=y_trainval,
        random_state=random_state,
    )
    return {
        "X": X,
        "y": y,
        "ids": ids,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_val": ids_val,
        "ids_test": ids_test,
    }


def build_preprocessors(
    num_cols: List[str], cat_cols: List[str]
) -> Tuple[ColumnTransformer, ColumnTransformer]:
    linear = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", make_one_hot_encoder(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tree = ColumnTransformer(
        [("num", "passthrough", num_cols), ("cat", make_one_hot_encoder(), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return linear, tree


def build_classifier_specs(
    linear_preprocess: ColumnTransformer,
    tree_preprocess: ColumnTransformer,
    feature_cols: List[str],
    scale_pos_weight: float,
    random_state: int,
    n_jobs: int,
    run_xgboost: bool,
    run_tree_baselines: bool = True,
) -> List[Tuple[str, Any, Dict[str, List[Any]], List[str]]]:
    specs: List[Tuple[str, Any, Dict[str, List[Any]], List[str]]] = [
        (
            "Logistic Regression balanced",
            Pipeline(
                [
                    ("preprocess", clone(linear_preprocess)),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=300,
                            solver="liblinear",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]},
            feature_cols,
        ),
    ]
    if run_tree_baselines:
        specs.extend(
            [
                (
                    "Random Forest balanced",
                    Pipeline(
                        [
                            ("preprocess", clone(tree_preprocess)),
                            (
                                "model",
                                RandomForestClassifier(
                                    class_weight="balanced_subsample",
                                    random_state=random_state,
                                    n_jobs=n_jobs,
                                ),
                            ),
                        ]
                    ),
                    {
                        "model__n_estimators": [100, 200, 500],
                        "model__max_depth": [None, 5, 10],
                        "model__min_samples_leaf": [1, 3, 5],
                        "model__max_features": ["sqrt", "log2", None],
                    },
                    feature_cols,
                ),
                (
                    "Extra Trees balanced",
                    Pipeline(
                        [
                            ("preprocess", clone(tree_preprocess)),
                            (
                                "model",
                                ExtraTreesClassifier(
                                    class_weight="balanced",
                                    random_state=random_state,
                                    n_jobs=n_jobs,
                                ),
                            ),
                        ]
                    ),
                    {
                        "model__n_estimators": [100, 200, 500],
                        "model__max_depth": [None, 5, 10],
                        "model__min_samples_leaf": [1, 3, 5],
                        "model__max_features": ["sqrt", "log2", None],
                    },
                    feature_cols,
                ),
            ]
        )
    if run_xgboost:
        global XGBClassifier
        try:
            from xgboost import XGBClassifier as _XGBClassifier  # type: ignore

            XGBClassifier = _XGBClassifier
            specs.append(
                (
                    "XGBoost weighted",
                    Pipeline(
                        [
                            ("preprocess", clone(tree_preprocess)),
                            (
                                "model",
                                XGBClassifier(
                                    objective="binary:logistic",
                                    eval_metric="aucpr",
                                    scale_pos_weight=scale_pos_weight,
                                    random_state=random_state,
                                    n_jobs=n_jobs,
                                ),
                            ),
                        ]
                    ),
                    {
                        "model__n_estimators": [10, 20],
                        "model__max_depth": [2, 3],
                        "model__learning_rate": [0.03, 0.05],
                        "model__subsample": [0.85],
                        "model__colsample_bytree": [0.85],
                        "model__reg_lambda": [5.0],
                        "model__min_child_weight": [3],
                    },
                    feature_cols,
                )
            )
        except Exception as exc:
            print(f"[M5] XGBoost skipped: {exc}")
    return specs


def build_smote_baseline(
    num_cols: List[str], random_state: int
) -> Tuple[str, Any, List[str]] | None:
    if not IMBLEARN_AVAILABLE:
        return None
    return (
        "SMOTETomek + Logistic Regression",
        ImbPipeline(
            [
                ("scale", StandardScaler()),
                ("resample", SMOTETomek(random_state=random_state)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000, solver="liblinear", random_state=random_state
                    ),
                ),
            ]
        ),
        num_cols,
    )


def tune_and_benchmark_classifiers(
    specs: List[Tuple[str, Any, Dict[str, List[Any]], List[str]]],
    split: Dict[str, Any],
    f_beta: float,
    output_dir: Path,
    random_state: int,
    cv_folds: int,
    tuning_n_iter: int,
    smote_baseline: Tuple[str, Any, List[str]] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[Any, List[str]]]]:
    results: List[Dict[str, Any]] = []
    tuning_rows: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Tuple[Any, List[str]]] = {}

    dummy = DummyClassifier(strategy="prior", random_state=random_state)
    dummy.fit(split["X_train"], split["y_train"])
    val_proba = dummy.predict_proba(split["X_val"])[:, 1]
    threshold, _ = best_fbeta_threshold(split["y_val"], val_proba, beta=f_beta)
    row: Dict[str, Any] = {
        "model": "Dummy prior",
        "tuned": False,
        "features_used": "none",
        "best_cv_PR_AUC": np.nan,
        "best_params": "{}",
    }
    for key, value in evaluate_proba(
        split["y_val"], val_proba, threshold, beta=f_beta
    ).items():
        row[f"val_{key}"] = value
    for key, value in evaluate_proba(
        split["y_test"],
        dummy.predict_proba(split["X_test"])[:, 1],
        threshold,
        beta=f_beta,
    ).items():
        row[f"test_{key}"] = value
    results.append(row)
    fitted_models["Dummy prior"] = (dummy, list(split["X"].columns))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for name, estimator, param_distributions, cols in specs:
        print(f"[M5] Tuning classifier: {name}")
        X_train = split["X_train"][cols].reset_index(drop=True)
        y_train = split["y_train"].reset_index(drop=True)
        if param_distributions:
            params_list = list(
                ParameterSampler(
                    param_distributions, n_iter=tuning_n_iter, random_state=random_state
                )
            )
            scored: List[Tuple[float, float, Dict[str, Any]]] = []
            for param_idx, params in enumerate(params_list, start=1):
                print(f"[M5]   {name}: CV candidate {param_idx}/{len(params_list)}")
                fold_scores = []
                for train_idx, valid_idx in cv.split(X_train, y_train):
                    candidate = clone(estimator)
                    candidate.set_params(**params)
                    candidate.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                    fold_scores.append(
                        float(
                            average_precision_score(
                                y_train.iloc[valid_idx],
                                candidate.predict_proba(X_train.iloc[valid_idx])[:, 1],
                            )
                        )
                    )
                scored.append(
                    (float(np.mean(fold_scores)), float(np.std(fold_scores)), params)
                )
            scored.sort(key=lambda x: x[0], reverse=True)
            best_mean, best_std, best_params = scored[0]
            tuned_flag = True
        else:
            # Fixed lightweight benchmark for non-champion tree baselines.
            # The final reporting model prioritizes calibrated probability quality;
            # tree models remain comparison baselines unless explicitly enabled/tuned.
            best_mean, best_std, best_params = np.nan, np.nan, {}
            scored = []
            tuned_flag = False
        best_model = clone(estimator)
        best_model.set_params(**best_params)
        best_model.fit(split["X_train"][cols], split["y_train"])
        fitted_models[name] = (best_model, cols)
        for rank, (mean_score, std_score, params) in enumerate(scored, start=1):
            tuning_rows.append(
                {
                    "model": name,
                    "rank_test_score": rank,
                    "mean_cv_PR_AUC": mean_score,
                    "std_cv_PR_AUC": std_score,
                    "params": json.dumps(params, sort_keys=True),
                }
            )
        val_proba = best_model.predict_proba(split["X_val"][cols])[:, 1]
        threshold, _ = best_fbeta_threshold(split["y_val"], val_proba, beta=f_beta)
        result: Dict[str, Any] = {
            "model": name,
            "tuned": tuned_flag,
            "features_used": "numeric+categorical",
            "best_cv_PR_AUC": best_mean,
            "best_cv_PR_AUC_std": best_std,
            "best_params": json.dumps(best_params, sort_keys=True),
        }
        for key, value in evaluate_proba(
            split["y_val"], val_proba, threshold, beta=f_beta
        ).items():
            result[f"val_{key}"] = value
        for key, value in evaluate_proba(
            split["y_test"],
            best_model.predict_proba(split["X_test"][cols])[:, 1],
            threshold,
            beta=f_beta,
        ).items():
            result[f"test_{key}"] = value
        results.append(result)

    if smote_baseline is not None:
        name, estimator, cols = smote_baseline
        print(f"[M5] Fitting optional classifier: {name}")
        estimator.fit(split["X_train"][cols], split["y_train"])
        fitted_models[name] = (estimator, cols)
        val_proba = estimator.predict_proba(split["X_val"][cols])[:, 1]
        threshold, _ = best_fbeta_threshold(split["y_val"], val_proba, beta=f_beta)
        result = {
            "model": name,
            "tuned": False,
            "features_used": "numeric_only",
            "best_cv_PR_AUC": np.nan,
            "best_params": "{}",
        }
        for key, value in evaluate_proba(
            split["y_val"], val_proba, threshold, beta=f_beta
        ).items():
            result[f"val_{key}"] = value
        for key, value in evaluate_proba(
            split["y_test"],
            estimator.predict_proba(split["X_test"][cols])[:, 1],
            threshold,
            beta=f_beta,
        ).items():
            result[f"test_{key}"] = value
        results.append(result)

    metrics = (
        pd.DataFrame(results)
        .sort_values(["val_PR_AUC", "val_F2_score"], ascending=False)
        .reset_index(drop=True)
    )
    tuning = pd.DataFrame(tuning_rows)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    tuning.to_csv(output_dir / "model_tuning_results.csv", index=False)
    metrics[
        [
            c
            for c in ["model", "best_cv_PR_AUC", "best_cv_PR_AUC_std", "best_params"]
            if c in metrics.columns
        ]
    ].to_csv(output_dir / "model_cv_summary.csv", index=False)
    return metrics, tuning, fitted_models


def calibrate_prefit_model(
    base_model: Any,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    methods: Iterable[str] = ("sigmoid", "isotonic"),
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"raw_uncalibrated": base_model}
    for method in methods:
        try:
            out[method] = PrefitProbabilityCalibrator(base_model, method=method).fit(
                X_calib, y_calib
            )
        except Exception as exc:
            print(f"[M5] Calibration method {method} failed: {exc}")
    return out


def select_calibrated_churn_model(
    metrics: pd.DataFrame,
    fitted_models: Dict[str, Tuple[Any, List[str]]],
    split: Dict[str, Any],
    output_dir: Path,
    f_beta: float,
) -> Tuple[str, str, float, Any, List[str], pd.DataFrame]:
    eligible = metrics[metrics["model"] != "Dummy prior"].copy()
    champion_name = str(eligible.iloc[0]["model"])
    base_model, cols = fitted_models[champion_name]
    candidates = calibrate_prefit_model(
        base_model, split["X_val"][cols], split["y_val"]
    )
    rows: List[Dict[str, Any]] = []
    for method, model in candidates.items():
        val_p = model.predict_proba(split["X_val"][cols])[:, 1]
        test_p = model.predict_proba(split["X_test"][cols])[:, 1]
        threshold, _ = best_fbeta_threshold(split["y_val"], val_p, beta=f_beta)
        row: Dict[str, Any] = {
            "champion_model": champion_name,
            "calibration_method": method,
        }
        for key, value in evaluate_proba(
            split["y_val"], val_p, threshold, beta=f_beta
        ).items():
            row[f"val_{key}"] = value
        for key, value in evaluate_proba(
            split["y_test"], test_p, threshold, beta=f_beta
        ).items():
            row[f"test_{key}"] = value
        rows.append(row)
    cal_summary = (
        pd.DataFrame(rows)
        .sort_values(["val_brier_score", "val_PR_AUC"], ascending=[True, False])
        .reset_index(drop=True)
    )
    selected = cal_summary.iloc[0]
    selected_method = str(selected["calibration_method"])
    selected_model = candidates[selected_method]
    threshold = float(selected["val_threshold"])
    cal_summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    cal_summary[cal_summary["calibration_method"] == selected_method].to_csv(
        output_dir / "champion_test_metrics.csv", index=False
    )

    val_dec = calibration_by_decile(
        split["y_val"], selected_model.predict_proba(split["X_val"][cols])[:, 1]
    )
    val_dec.insert(0, "dataset", "validation")
    test_dec = calibration_by_decile(
        split["y_test"], selected_model.predict_proba(split["X_test"][cols])[:, 1]
    )
    test_dec.insert(0, "dataset", "test")
    dec = pd.concat([val_dec, test_dec], ignore_index=True)
    dec.insert(0, "calibration_method", selected_method)
    dec.to_csv(output_dir / "calibration_by_decile.csv", index=False)
    return champion_name, selected_method, threshold, selected_model, cols, cal_summary


def feature_business_interpretation(feature: str) -> str:
    f = feature.lower()
    if "recency" in f or "inactive" in f or "days_since" in f:
        return "Recent inactivity or time since engagement; usually linked with churn risk."
    if "frequency" in f or "freq" in f:
        return "Purchase frequency / activity intensity."
    if "monetary" in f or "basket" in f or "sales" in f or "value" in f:
        return "Customer spending level or basket value."
    if "promo" in f or "coupon" in f or "campaign" in f:
        return "Promotion, coupon, or campaign responsiveness."
    if "brand" in f:
        return "Brand preference pattern."
    if "slope" in f or "trend" in f:
        return "Behavioral trend over time."
    if "store" in f:
        return "Primary store/location pattern encoded as categorical information."
    return "Predictive association used by the model; not a causal claim."


def export_feature_importance(
    champion_model: Any, champion_cols: List[str], output_dir: Path
) -> pd.DataFrame:
    model_for_importance = getattr(champion_model, "base_model", champion_model)
    importance = extract_feature_importance(model_for_importance, champion_cols)
    if not importance.empty:
        importance["business_interpretation"] = importance["feature"].map(
            feature_business_interpretation
        )
        importance["causality_warning"] = (
            "Predictive association only; causal effect requires A/B testing."
        )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    return importance


def build_discounted_value_labels(
    features: pd.DataFrame,
    paths: Dict[str, Path],
    id_col: str,
    cut_off_day: int,
    prediction_horizon_days: int,
    annual_discount_rate: float,
) -> pd.DataFrame:
    txn_path = paths["transaction_master_parquet"]
    if not txn_path.exists():
        raise FileNotFoundError(
            f"Transaction file not found for value labels: {txn_path}"
        )
    txns = pd.read_parquet(txn_path, columns=[id_col, "DAY", "SALES_VALUE"])
    txns[id_col] = txns[id_col].astype(int)
    max_day = int(txns["DAY"].max())
    horizon_end = min(cut_off_day + prediction_horizon_days, max_day)
    future = txns[(txns["DAY"] >= cut_off_day) & (txns["DAY"] <= horizon_end)].copy()
    if future.empty:
        raise ValueError("No post-cutoff transactions found for value labels.")
    future["days_after_cutoff"] = np.maximum(future["DAY"] - cut_off_day, 0)
    future["discount_factor"] = np.power(
        1.0 + annual_discount_rate, future["days_after_cutoff"] / 365.0
    )
    future["discounted_sales_value"] = future["SALES_VALUE"] / future["discount_factor"]
    labels = future.groupby(id_col, as_index=False).agg(
        future_revenue_60d=("SALES_VALUE", "sum"),
        discounted_future_revenue_60d=("discounted_sales_value", "sum"),
        future_txn_count_60d=("DAY", "size"),
        first_future_day=("DAY", "min"),
        last_future_day=("DAY", "max"),
    )
    out = features.merge(labels, on=id_col, how="left")
    for col in [
        "future_revenue_60d",
        "discounted_future_revenue_60d",
        "future_txn_count_60d",
    ]:
        out[col] = out[col].fillna(0.0)
    out["future_active_flag"] = (out["future_revenue_60d"] > 0).astype(int)
    out["annual_discount_rate"] = annual_discount_rate
    out["value_horizon_start_day"] = cut_off_day
    out["value_horizon_end_day"] = horizon_end
    out.to_csv(paths["models_dir"] / "discounted_value_labels.csv", index=False)
    # Keep backward-compatible filename for previous readers.
    return out


def build_value_regressor_specs(
    linear_preprocess: ColumnTransformer,
    tree_preprocess: ColumnTransformer,
    random_state: int,
    n_estimators: int,
    n_jobs: int,
    run_tree: bool,
    run_xgboost: bool,
) -> List[Tuple[str, Any]]:
    specs: List[Tuple[str, Any]] = [
        (
            "Ridge Regression",
            Pipeline(
                [("preprocess", clone(linear_preprocess)), ("model", Ridge(alpha=1.0))]
            ),
        )
    ]
    if run_tree:
        specs.append(
            (
                "Random Forest Regressor",
                Pipeline(
                    [
                        ("preprocess", clone(tree_preprocess)),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=n_estimators,
                                min_samples_leaf=5,
                                random_state=random_state,
                                n_jobs=n_jobs,
                            ),
                        ),
                    ]
                ),
            )
        )
    if run_xgboost:
        global XGBRegressor
        try:
            from xgboost import XGBRegressor as _XGBRegressor  # type: ignore

            XGBRegressor = _XGBRegressor
            specs.append(
                (
                    "XGBoost Regressor",
                    Pipeline(
                        [
                            ("preprocess", clone(tree_preprocess)),
                            (
                                "model",
                                XGBRegressor(
                                    n_estimators=n_estimators,
                                    max_depth=3,
                                    learning_rate=0.03,
                                    subsample=0.85,
                                    colsample_bytree=0.85,
                                    objective="reg:squarederror",
                                    reg_lambda=5.0,
                                    random_state=random_state,
                                    n_jobs=n_jobs,
                                ),
                            ),
                        ]
                    ),
                )
            )
        except Exception as exc:
            print(f"[M5] XGBoost regressor skipped: {exc}")
    return specs


def train_two_part_value_model(
    labels: pd.DataFrame,
    split: Dict[str, Any],
    feature_cols: List[str],
    linear_preprocess: ColumnTransformer,
    tree_preprocess: ColumnTransformer,
    output_dir: Path,
    id_col: str,
    random_state: int,
    f_beta: float,
    n_estimators: int,
    n_jobs: int,
    run_tree_value_models: bool,
    run_xgboost: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, str, Any, str]:
    id_sets = {
        name: set(split[f"ids_{name}"].astype(int)) for name in ["train", "val", "test"]
    }
    train = labels[labels[id_col].isin(id_sets["train"])].copy()
    val = labels[labels[id_col].isin(id_sets["val"])].copy()
    test = labels[labels[id_col].isin(id_sets["test"])].copy()

    # Part 1: probability of future activity.
    active_candidates: List[Tuple[str, Any]] = [
        (
            "Active Logistic Regression",
            Pipeline(
                [
                    ("preprocess", clone(linear_preprocess)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=300, solver="liblinear", random_state=random_state
                        ),
                    ),
                ]
            ),
        ),
    ]
    if run_tree_value_models:
        active_candidates.append(
            (
                "Active Random Forest",
                Pipeline(
                    [
                        ("preprocess", clone(tree_preprocess)),
                        (
                            "model",
                            RandomForestClassifier(
                                n_estimators=n_estimators,
                                min_samples_leaf=5,
                                random_state=random_state,
                                n_jobs=n_jobs,
                            ),
                        ),
                    ]
                ),
            )
        )
    active_rows: List[Dict[str, Any]] = []
    active_models: Dict[str, Any] = {}
    for name, model in active_candidates:
        print(f"[M5] Fitting future-active model: {name}")
        model.fit(train[feature_cols], train["future_active_flag"])
        calibrated_candidates = calibrate_prefit_model(
            model,
            val[feature_cols],
            val["future_active_flag"],
            methods=("sigmoid", "isotonic"),
        )
        for method, cal_model in calibrated_candidates.items():
            val_p = cal_model.predict_proba(val[feature_cols])[:, 1]
            test_p = cal_model.predict_proba(test[feature_cols])[:, 1]
            threshold, _ = best_fbeta_threshold(
                val["future_active_flag"], val_p, beta=f_beta
            )
            row: Dict[str, Any] = {"active_model": name, "calibration_method": method}
            for key, value in evaluate_proba(
                val["future_active_flag"], val_p, threshold, beta=f_beta
            ).items():
                row[f"val_{key}"] = value
            for key, value in evaluate_proba(
                test["future_active_flag"], test_p, threshold, beta=f_beta
            ).items():
                row[f"test_{key}"] = value
            active_rows.append(row)
            active_models[f"{name}__{method}"] = cal_model
    active_metrics = (
        pd.DataFrame(active_rows)
        .sort_values(["val_brier_score", "val_PR_AUC"], ascending=[True, False])
        .reset_index(drop=True)
    )
    active_metrics.to_csv(output_dir / "active_model_metrics.csv", index=False)
    active_key = f"{active_metrics.iloc[0]['active_model']}__{active_metrics.iloc[0]['calibration_method']}"
    active_champion = active_models[active_key]

    # Part 2: value conditional on being future-active.
    train_pos = train[train["future_active_flag"] == 1].copy()
    val_pos = val[val["future_active_flag"] == 1].copy()
    test_pos = test[test["future_active_flag"] == 1].copy()
    if train_pos.empty:
        raise ValueError(
            "No future-active customers in training split; cannot train conditional value model."
        )
    y_train = np.log1p(train_pos["discounted_future_revenue_60d"])
    y_val = np.log1p(val_pos["discounted_future_revenue_60d"])
    y_test = np.log1p(test_pos["discounted_future_revenue_60d"])
    reg_specs = build_value_regressor_specs(
        linear_preprocess,
        tree_preprocess,
        random_state,
        n_estimators,
        n_jobs,
        run_tree_value_models,
        run_xgboost,
    )
    value_rows: List[Dict[str, Any]] = []
    value_models: Dict[str, Any] = {}
    for name, reg in reg_specs:
        print(f"[M5] Fitting conditional discounted value model: {name}")
        reg.fit(train_pos[feature_cols], y_train)
        value_models[name] = reg
        val_pred = reg.predict(val_pos[feature_cols]) if len(val_pos) else np.array([])
        test_pred = (
            reg.predict(test_pos[feature_cols]) if len(test_pos) else np.array([])
        )
        row: Dict[str, Any] = {
            "value_model": name,
            "target": "log1p(discounted_future_revenue_60d | active)",
        }
        row.update(regression_metrics(y_val, val_pred, "val"))
        row.update(regression_metrics(y_test, test_pred, "test"))
        value_rows.append(row)
    value_metrics = (
        pd.DataFrame(value_rows)
        .sort_values("val_RMSE_log", ascending=True)
        .reset_index(drop=True)
    )
    value_metrics.to_csv(output_dir / "value_model_metrics.csv", index=False)
    value_name = str(value_metrics.iloc[0]["value_model"])
    return (
        active_metrics,
        value_metrics,
        active_champion,
        active_key,
        value_models[value_name],
        value_name,
    )


def export_active_churn_overlap_audit(
    pred: pd.DataFrame, paths: Dict[str, Path]
) -> None:
    """Audit whether the value-model active target is effectively the inverse of churn.

    This is diagnostic only. It helps avoid overclaiming that the active-stage
    target is an independent business target when it may be very close to the
    operational non-churn label.
    """
    required = {"actual_churn_flag", "future_active_flag"}
    if not required.issubset(pred.columns):
        pd.DataFrame(
            [
                {
                    "status": "skipped_missing_required_columns",
                    "required_columns": ",".join(sorted(required)),
                }
            ]
        ).to_csv(
            paths["models_dir"] / "active_churn_target_overlap_audit.csv", index=False
        )
        return

    tmp = pred[["actual_churn_flag", "future_active_flag"]].dropna().copy()
    if tmp.empty:
        pd.DataFrame([{"status": "skipped_no_non_missing_rows"}]).to_csv(
            paths["models_dir"] / "active_churn_target_overlap_audit.csv", index=False
        )
        return

    tmp["actual_churn_flag"] = tmp["actual_churn_flag"].astype(int)
    tmp["future_active_flag"] = tmp["future_active_flag"].astype(int)
    tmp["non_churn_flag"] = 1 - tmp["actual_churn_flag"]
    tmp["active_equals_non_churn"] = tmp["future_active_flag"] == tmp["non_churn_flag"]

    corr = np.nan
    if tmp["future_active_flag"].nunique() > 1 and tmp["non_churn_flag"].nunique() > 1:
        corr = float(tmp["future_active_flag"].corr(tmp["non_churn_flag"]))

    rows = [
        {
            "status": "completed",
            "n_rows": int(len(tmp)),
            "future_active_rate": float(tmp["future_active_flag"].mean()),
            "non_churn_rate": float(tmp["non_churn_flag"].mean()),
            "match_rate_future_active_vs_non_churn": float(
                tmp["active_equals_non_churn"].mean()
            ),
            "correlation_future_active_vs_non_churn": corr,
            "active_1_non_churn_1": int(
                ((tmp["future_active_flag"] == 1) & (tmp["non_churn_flag"] == 1)).sum()
            ),
            "active_1_non_churn_0": int(
                ((tmp["future_active_flag"] == 1) & (tmp["non_churn_flag"] == 0)).sum()
            ),
            "active_0_non_churn_1": int(
                ((tmp["future_active_flag"] == 0) & (tmp["non_churn_flag"] == 1)).sum()
            ),
            "active_0_non_churn_0": int(
                ((tmp["future_active_flag"] == 0) & (tmp["non_churn_flag"] == 0)).sum()
            ),
            "interpretation": "High overlap means p_future_active is close to the inverse churn label and should not be treated as an independent targeting signal.",
        }
    ]
    pd.DataFrame(rows).to_csv(
        paths["models_dir"] / "active_churn_target_overlap_audit.csv", index=False
    )


def calculate_deciles_and_segments(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["risk_decile"] = pd.qcut(
        pred["p_churn_calibrated"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)
    pred["value_decile"] = pd.qcut(
        pred["predicted_expected_discounted_value_60d"].rank(
            method="first", ascending=False
        ),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)
    high_risk = pred["risk_decile"] <= 3
    high_value = pred["value_decile"] <= 3
    pred["priority_segment"] = np.select(
        [high_risk & high_value, high_risk & ~high_value, ~high_risk & high_value],
        ["High Risk - High Value", "High Risk - Low Value", "Low Risk - High Value"],
        default="Low Risk - Low Value",
    )
    return pred


def make_scenario_grid(configured: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, p in configured.items():
        rows.append(
            {
                "scenario": name,
                "gross_margin": float(p["gross_margin"]),
                "save_rate_given_treatment": float(
                    p.get("save_rate_given_treatment", p.get("retention_lift", 0.0))
                ),
                "treatment_cost": float(p["treatment_cost"]),
                "scenario_type": "named",
            }
        )
    for margin in [0.20, 0.25, 0.30, 0.40]:
        for save_rate in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            for cost in [1.0, 2.0, 3.0, 5.0, 8.0]:
                rows.append(
                    {
                        "scenario": f"m{margin:.2f}_s{save_rate:.2f}_c{cost:.1f}",
                        "gross_margin": margin,
                        "save_rate_given_treatment": save_rate,
                        "treatment_cost": cost,
                        "scenario_type": "sensitivity",
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(
        ["gross_margin", "save_rate_given_treatment", "treatment_cost", "scenario_type"]
    )


def apply_expected_profit(
    pred: pd.DataFrame, scenario_grid: pd.DataFrame, paths: Dict[str, Path], id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    summary_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    for _, p in scenario_grid.iterrows():
        scenario = str(p["scenario"])
        margin = float(p["gross_margin"])
        save_rate = float(p["save_rate_given_treatment"])
        cost = float(p["treatment_cost"])
        ep = (
            pred["p_churn_calibrated"]
            * save_rate
            * pred["predicted_discounted_value_60d_if_active"]
            * margin
            - cost
        )
        profitable = ep > 0
        row = {
            "scenario": scenario,
            "scenario_type": p["scenario_type"],
            "gross_margin": margin,
            "save_rate_given_treatment": save_rate,
            "treatment_cost": cost,
            "profitable_customer_count": int(profitable.sum()),
            "profitable_customer_share": float(profitable.mean()),
            "total_expected_incremental_profit_if_target_positive_only": float(
                ep[profitable].sum()
            )
            if profitable.any()
            else 0.0,
            "top_30pct_risk_customer_count": int((pred["risk_decile"] <= 3).sum()),
            "total_expected_incremental_profit_if_target_top_30pct_risk": float(
                ep[pred["risk_decile"] <= 3].sum()
            ),
        }
        if p["scenario_type"] == "named":
            pred[f"expected_incremental_profit_{scenario}"] = ep
            pred[f"profitable_to_treat_{scenario}"] = profitable
            pred[f"predicted_discounted_margin_60d_if_active_{scenario}"] = (
                pred["predicted_discounted_value_60d_if_active"] * margin
            )
            pred[f"predicted_expected_discounted_margin_60d_{scenario}"] = (
                pred["predicted_expected_discounted_value_60d"] * margin
            )
            summary_rows.append(row)
        else:
            sensitivity_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    summary.to_csv(paths["models_dir"] / "scenario_profit_summary.csv", index=False)
    sensitivity.to_csv(
        paths["models_dir"] / "scenario_sensitivity_grid.csv", index=False
    )
    break_even = pred[
        [
            id_col,
            "p_churn_calibrated",
            "predicted_discounted_value_60d_if_active",
            "predicted_expected_discounted_value_60d",
        ]
    ].copy()
    for _, p in summary.iterrows():
        scenario = str(p["scenario"])
        break_even[f"break_even_treatment_cost_{scenario}"] = (
            break_even["p_churn_calibrated"]
            * float(p["save_rate_given_treatment"])
            * break_even["predicted_discounted_value_60d_if_active"]
            * float(p["gross_margin"])
        )
    break_even.to_csv(paths["models_dir"] / "break_even_analysis.csv", index=False)
    return pred, summary, sensitivity


def export_ranking_and_profit_diagnostics(
    pred: pd.DataFrame,
    paths: Dict[str, Path],
    top_k_shares: Iterable[float],
    decile_count: int,
) -> None:
    """Export ranking-quality and profit-threshold diagnostics for business review."""
    score_specs: List[Tuple[str, str]] = [("p_churn_calibrated", "p_churn_calibrated")]
    if "risk_value_score" in pred.columns:
        score_specs.append(("risk_value_score", "risk_value_score"))
    if "expected_incremental_profit_base" in pred.columns:
        score_specs.append(
            ("expected_incremental_profit_base", "expected_incremental_profit_base")
        )

    decile_tables: List[pd.DataFrame] = []
    topk_tables: List[pd.DataFrame] = []
    for col, score_name in score_specs:
        if col not in pred.columns:
            continue
        decile = ranking_decile_performance(
            pred["actual_churn_flag"],
            pred[col],
            n_deciles=decile_count,
            score_name=score_name,
        )
        if not decile.empty:
            decile.insert(0, "ranking_score", score_name)
            decile_tables.append(decile)
        topk = top_k_precision_summary(
            pred["actual_churn_flag"],
            pred[col],
            top_k_shares=top_k_shares,
            score_name=score_name,
        )
        if not topk.empty:
            topk_tables.append(topk)

    if decile_tables:
        pd.concat(decile_tables, ignore_index=True).to_csv(
            paths["models_dir"] / "ranking_decile_performance.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(
            paths["models_dir"] / "ranking_decile_performance.csv", index=False
        )
    if topk_tables:
        pd.concat(topk_tables, ignore_index=True).to_csv(
            paths["models_dir"] / "top_k_precision_summary.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(
            paths["models_dir"] / "top_k_precision_summary.csv", index=False
        )

    expected_profit_cols = [
        c for c in pred.columns if c.startswith("expected_incremental_profit_")
    ]
    profit = profit_threshold_analysis(
        pred,
        expected_profit_cols=expected_profit_cols,
        y_col="actual_churn_flag",
        p_col="p_churn_calibrated",
        top_k_shares=top_k_shares,
        min_profit=0.0,
    )
    profit.to_csv(paths["models_dir"] / "profit_threshold_analysis.csv", index=False)


def _recommendation_risk_from_rule_group(group: Any, trivial: Any = None) -> str:
    """Map MBA rule metadata to a conservative treatment-design risk label."""
    if pd.notna(trivial):
        try:
            if bool(trivial):
                return "high_trivial_or_discount_leakage_risk"
        except Exception:
            pass
    if pd.isna(group):
        return "unknown"
    group_str = str(group).strip().upper()
    if group_str.startswith("A"):
        return "low"
    if group_str.startswith("B"):
        return "medium_manual_review"
    if group_str.startswith("C"):
        return "high_trivial_or_discount_leakage_risk"
    return "unknown"


def merge_recommendation_metadata(
    pred: pd.DataFrame,
    paths: Dict[str, Path],
    id_col: str,
    top_n: int = 3,
) -> pd.DataFrame:
    """Merge M3 MBA/RecSys outputs as treatment metadata, not model features."""
    pred = pred.copy()
    voucher_path = paths.get("voucher_recommendations_csv")
    rules_path = paths.get("final_campaign_rules_csv")
    audit: Dict[str, Any] = {
        "voucher_file_exists": bool(voucher_path and voucher_path.exists()),
        "mba_rules_file_exists": bool(rules_path and rules_path.exists()),
        "m5_customer_rows": int(len(pred)),
        "top_n_vouchers_requested": int(top_n),
        "recommendation_role": "treatment_metadata_only_not_training_feature",
    }

    if not voucher_path or not voucher_path.exists():
        pred["offer_available_flag"] = False
        pred["recommendation_source"] = "none_voucher_file_missing"
        pd.DataFrame([audit]).to_csv(
            paths["models_dir"] / "recommendation_merge_audit.csv", index=False
        )
        return pred

    vouchers = pd.read_csv(voucher_path)
    audit["voucher_rows"] = int(len(vouchers))
    if id_col not in vouchers.columns:
        audit["merge_status"] = f"skipped_missing_{id_col}"
        pred["offer_available_flag"] = False
        pred["recommendation_source"] = "none_missing_household_key"
        pd.DataFrame([audit]).to_csv(
            paths["models_dir"] / "recommendation_merge_audit.csv", index=False
        )
        return pred

    vouchers[id_col] = vouchers[id_col].astype(int)
    rank_col = (
        "voucher_recommendation_rank"
        if "voucher_recommendation_rank" in vouchers.columns
        else "recommendation_rank"
    )
    if rank_col not in vouchers.columns:
        vouchers[rank_col] = vouchers.groupby(id_col).cumcount() + 1
    vouchers[rank_col] = vouchers[rank_col].astype(int)
    vouchers = (
        vouchers.sort_values([id_col, rank_col]).groupby(id_col).head(top_n).copy()
    )
    audit["voucher_households"] = int(vouchers[id_col].nunique())

    # Optional MBA rule metadata by recommended consequent. This is used only for
    # treatment-design risk labelling, not for model training.
    rule_lookup = pd.DataFrame()
    if rules_path and rules_path.exists():
        try:
            rules = pd.read_csv(rules_path)
            if "consequents_str" in rules.columns:
                sort_cols = [
                    c
                    for c in ["business_score", "lift", "confidence", "basket_count"]
                    if c in rules.columns
                ]
                if sort_cols:
                    rules = rules.sort_values(sort_cols, ascending=False)
                keep = [
                    c
                    for c in [
                        "consequents_str",
                        "rule_group",
                        "trivial_flag",
                        "business_score",
                        "recommended_campaign_use",
                    ]
                    if c in rules.columns
                ]
                rule_lookup = (
                    rules[keep]
                    .drop_duplicates("consequents_str")
                    .rename(columns={"consequents_str": "recommended_item"})
                )
        except Exception as exc:
            audit["mba_rules_merge_error"] = str(exc)

    if not rule_lookup.empty and "recommended_item" in vouchers.columns:
        vouchers = vouchers.merge(rule_lookup, on="recommended_item", how="left")
        vouchers["discount_leakage_risk"] = vouchers.apply(
            lambda r: _recommendation_risk_from_rule_group(
                r.get("rule_group"), r.get("trivial_flag")
            ),
            axis=1,
        )
    else:
        vouchers["rule_group"] = "unknown"
        vouchers["discount_leakage_risk"] = "unknown"

    value_cols = [
        "recommended_item",
        "recommended_item_group",
        "predicted_purchase_score",
        "strongest_purchased_anchor_item",
        "strongest_purchased_anchor_group",
        "anchor_item_similarity",
        "score_method",
        "rule_group",
        "discount_leakage_risk",
    ]
    value_cols = [c for c in value_cols if c in vouchers.columns]
    wide = vouchers[[id_col, rank_col, *value_cols]].copy()
    pieces = []
    for r in range(1, top_n + 1):
        one = wide[wide[rank_col] == r].drop(columns=[rank_col]).copy()
        rename = {c: f"{c}_{r}" for c in value_cols}
        one = one.rename(columns=rename)
        pieces.append(one)
    if pieces:
        rec_wide = pieces[0]
        for piece in pieces[1:]:
            rec_wide = rec_wide.merge(piece, on=id_col, how="outer")
        pred = pred.merge(rec_wide, on=id_col, how="left")

    pred["offer_available_flag"] = pred.get(
        "recommended_item_1", pd.Series(index=pred.index, dtype=object)
    ).notna()
    pred["recommendation_source"] = np.where(
        pred["offer_available_flag"],
        "M3_RecSys_personalized_voucher",
        "none_no_household_offer",
    )
    if "discount_leakage_risk_1" in pred.columns:
        pred["primary_offer_discount_leakage_risk"] = pred[
            "discount_leakage_risk_1"
        ].fillna("unknown")
    else:
        pred["primary_offer_discount_leakage_risk"] = "unknown"

    high_risk = (
        pred["risk_decile"] <= 3
        if "risk_decile" in pred.columns
        else pd.Series(False, index=pred.index)
    )
    audit.update(
        {
            "merge_status": "completed",
            "m5_customers_with_offer": int(pred["offer_available_flag"].sum()),
            "m5_offer_coverage_share": float(pred["offer_available_flag"].mean()),
            "high_risk_customers": int(high_risk.sum()),
            "high_risk_customers_with_offer": int(
                (high_risk & pred["offer_available_flag"]).sum()
            ),
            "high_risk_offer_coverage_share": float(
                (high_risk & pred["offer_available_flag"]).sum()
                / max(high_risk.sum(), 1)
            ),
            "top_offer_low_risk_count": int(
                (
                    pred.get("primary_offer_discount_leakage_risk", "unknown") == "low"
                ).sum()
            )
            if "primary_offer_discount_leakage_risk" in pred.columns
            else 0,
            "top_offer_medium_review_count": int(
                (
                    pred.get("primary_offer_discount_leakage_risk", "unknown")
                    == "medium_manual_review"
                ).sum()
            )
            if "primary_offer_discount_leakage_risk" in pred.columns
            else 0,
            "top_offer_high_risk_count": int(
                (
                    pred.get("primary_offer_discount_leakage_risk", "unknown")
                    == "high_trivial_or_discount_leakage_risk"
                ).sum()
            )
            if "primary_offer_discount_leakage_risk" in pred.columns
            else 0,
            "note": "Recommendations are merged after scoring as offer metadata. They are not used as churn-model training features.",
        }
    )
    pd.DataFrame([audit]).to_csv(
        paths["models_dir"] / "recommendation_merge_audit.csv", index=False
    )
    return pred


def score_customers(
    features: pd.DataFrame,
    value_labels: pd.DataFrame,
    churn_model: Any,
    churn_cols: List[str],
    churn_threshold: float,
    active_model: Any,
    value_model: Any,
    value_model_name: str,
    feature_cols: List[str],
    id_col: str,
    target_col: str,
    scenarios: Dict[str, Dict[str, float]],
    paths: Dict[str, Path],
    cut_off_day: int,
    discount_rate: float,
    active_model_name: str,
    champion_name: str,
    calibration_method: str,
    top_k_shares: Iterable[float],
    decile_count: int,
    recommendation_top_n: int,
) -> pd.DataFrame:
    X = features[feature_cols].copy()
    y = features[target_col].astype(int).copy()
    p_churn = churn_model.predict_proba(features[churn_cols])[:, 1]
    # Raw probability from calibrated wrapper's base model, if available.
    base = getattr(churn_model, "base_model", churn_model)
    try:
        p_raw = base.predict_proba(features[churn_cols])[:, 1]
    except Exception:
        p_raw = p_churn
    p_active = active_model.predict_proba(X)[:, 1]
    value_log = value_model.predict(X)
    value_if_active = np.expm1(np.maximum(value_log, 0))
    expected_value = p_active * value_if_active
    predicted_churn = (p_churn >= churn_threshold).astype(int)
    pred = pd.DataFrame(
        {
            id_col: features[id_col].astype(int),
            "actual_churn_flag": y,
            "p_churn_raw": p_raw,
            "p_churn_calibrated": p_churn,
            "predicted_churn": predicted_churn,
            "p_future_active": p_active,
            "predicted_discounted_value_60d_if_active": value_if_active,
            "predicted_expected_discounted_value_60d": expected_value,
            "annual_discount_rate": discount_rate,
        }
    )
    # Include observed labels for audit only, not as model inputs.
    label_cols = [
        id_col,
        "future_active_flag",
        "future_revenue_60d",
        "discounted_future_revenue_60d",
    ]
    pred = pred.merge(value_labels[label_cols], on=id_col, how="left")
    pred = calculate_deciles_and_segments(pred)
    # Risk/value scores for experimental candidate selection. These do not assume
    # positive ROI under the base scenario; they identify households worth testing.
    pred["risk_rank"] = (
        pred["p_churn_calibrated"].rank(method="first", ascending=False).astype(int)
    )
    pred["risk_value_score"] = (
        pred["p_churn_calibrated"] * pred["predicted_discounted_value_60d_if_active"]
    )
    pred["risk_value_rank"] = (
        pred["risk_value_score"].rank(method="first", ascending=False).astype(int)
    )

    pred, _summary, _sens = apply_expected_profit(
        pred, make_scenario_grid(scenarios), paths, id_col
    )
    base_col = "expected_incremental_profit_base"
    base_profitable_col = "profitable_to_treat_base"
    pred["profit_rank_base"] = (
        pred[base_col].rank(method="first", ascending=False).astype(int)
    )
    base_positive_count = (
        int(pred[base_profitable_col].sum())
        if base_profitable_col in pred.columns
        else 0
    )
    if base_positive_count > 0:
        pred["priority_rank"] = pred["profit_rank_base"]
        pred["priority_rank_source"] = "profit_rank_base_positive_profit_available"
    else:
        pred["priority_rank"] = pred["risk_rank"]
        pred["priority_rank_source"] = "risk_rank_fallback_no_positive_base_profit"
    pred["recommended_treatment_action_base"] = np.where(
        pred[base_profitable_col],
        "Candidate for treatment under base scenario",
        np.where(
            pred["risk_decile"] <= 3,
            "A/B test candidate only; not profitable under base assumptions",
            "Monitor / no paid treatment",
        ),
    )
    pred["business_decision_note_base"] = np.where(
        base_positive_count > 0,
        "Base scenario has positive-profit candidates; profit rank can be used for treatment prioritization.",
        "No positive expected profit under base scenario; use risk/risk-value ranking for A/B test candidate selection, not auto-targeting.",
    )
    export_active_churn_overlap_audit(pred, paths)
    export_ranking_and_profit_diagnostics(
        pred, paths, top_k_shares=top_k_shares, decile_count=decile_count
    )
    pred = merge_recommendation_metadata(
        pred, paths, id_col=id_col, top_n=recommendation_top_n
    )
    pred = pred.sort_values("priority_rank").reset_index(drop=True)
    pred.to_csv(paths["models_dir"] / "churn_predictions.csv", index=False)
    pred[pred["risk_decile"] <= 3].sort_values(["risk_rank", "risk_value_rank"]).to_csv(
        paths["models_dir"] / "high_risk_customers_for_ab_test.csv", index=False
    )
    pred.sort_values(["priority_rank", "risk_value_rank"]).to_csv(
        paths["models_dir"] / "priority_customers_all.csv", index=False
    )

    package = {
        "version_note": "v3_hotfix_profit_ranking",
        "cut_off_day": cut_off_day,
        "feature_cols": feature_cols,
        "champion_churn_model_name": champion_name,
        "calibration_method": calibration_method,
        "champion_threshold_on_calibrated_probability": churn_threshold,
        "champion_churn_model_for_future_scoring": churn_model,
        "active_model_name": active_model_name,
        "active_model": active_model,
        "conditional_discounted_value_model_name": value_model_name,
        "conditional_discounted_value_model": value_model,
        "annual_discount_rate": discount_rate,
        "expected_profit_formula": "p_churn_calibrated * save_rate_given_treatment * predicted_discounted_value_60d_if_active * gross_margin - treatment_cost",
        "note": "M5 predicts calibrated risk and value for A/B test candidate selection. If base expected profit is non-positive for all customers, priority_rank falls back to risk ranking rather than profit ranking.",
    }
    joblib.dump(package, paths["models_dir"] / "model.pkl")
    return pred


def export_shap_outputs(
    churn_model: Any,
    champion_cols: List[str],
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    paths: Dict[str, Path],
    sample_size: int,
    top_n_customers: int,
) -> pd.DataFrame:
    status_path = paths["reports_internal_dir"] / "M5_shap_status.json"
    if not SHAP_AVAILABLE:
        status_path.write_text(
            json.dumps(
                {
                    "status": "skipped",
                    "reason": globals().get("SHAP_IMPORT_ERROR", "shap not available"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return pd.DataFrame()
    try:
        model = getattr(churn_model, "base_model", churn_model)
        if (
            not hasattr(model, "named_steps")
            or "preprocess" not in model.named_steps
            or "model" not in model.named_steps
        ):
            raise ValueError(
                "SHAP exporter currently supports sklearn Pipeline with preprocess/model steps."
            )
        preprocess = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        X_all = features[champion_cols].copy()
        n_sample = min(sample_size, len(X_all))
        bg = X_all.sample(n=min(200, n_sample), random_state=42)
        sample = X_all.sample(n=n_sample, random_state=43)
        X_bg_t = preprocess.transform(bg)
        X_sample_t = preprocess.transform(sample)
        try:
            names = [str(x) for x in preprocess.get_feature_names_out()]
        except Exception:
            names = [f"feature_{i}" for i in range(X_sample_t.shape[1])]
        if hasattr(estimator, "coef_"):
            explainer = shap.LinearExplainer(estimator, X_bg_t, feature_names=names)
            explanation = explainer(X_sample_t)
            shap_values = np.asarray(explanation.values)
        elif hasattr(estimator, "feature_importances_"):
            explainer = shap.TreeExplainer(estimator)
            explanation = explainer(X_sample_t)
            shap_values = np.asarray(explanation.values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        else:
            raise ValueError(
                f"Unsupported estimator type for fast SHAP: {type(estimator)}"
            )
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        global_importance = pd.DataFrame(
            {
                "feature": names[: shap_values.shape[1]],
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        )
        global_importance["business_interpretation"] = global_importance["feature"].map(
            feature_business_interpretation
        )
        global_importance["causality_warning"] = (
            "SHAP explains prediction contribution, not causal treatment effect."
        )
        global_importance = global_importance.sort_values(
            "mean_abs_shap", ascending=False
        ).reset_index(drop=True)
        global_importance.to_csv(
            paths["models_dir"] / "shap_global_importance.csv", index=False
        )

        # Top-risk local explanations.
        top_ids = (
            predictions.sort_values("p_churn_calibrated", ascending=False)
            .head(top_n_customers)["household_key"]
            .astype(int)
            .tolist()
        )
        id_to_row = {
            int(row["household_key"]): i
            for i, row in features.reset_index(drop=True).iterrows()
        }
        rows = []
        for hh in top_ids:
            if hh not in id_to_row:
                continue
            x_orig = features.iloc[[id_to_row[hh]]][champion_cols]
            x_t = preprocess.transform(x_orig)
            if hasattr(estimator, "coef_"):
                ev = explainer(x_t)
                vals = np.asarray(ev.values)
            else:
                ev = explainer(x_t)
                vals = np.asarray(ev.values)
                if vals.ndim == 3:
                    vals = vals[:, :, 1]
            vals = vals.reshape(-1)[: len(names)]
            order_pos = np.argsort(vals)[::-1]
            order_neg = np.argsort(vals)
            pred_row = predictions[predictions["household_key"] == hh].iloc[0]
            row = {
                "household_key": hh,
                "p_churn_calibrated": float(pred_row["p_churn_calibrated"]),
                "priority_segment": pred_row["priority_segment"],
            }
            for k in range(3):
                row[f"top_positive_driver_{k + 1}"] = names[order_pos[k]]
                row[f"top_positive_shap_{k + 1}"] = float(vals[order_pos[k]])
                row[f"top_negative_driver_{k + 1}"] = names[order_neg[k]]
                row[f"top_negative_shap_{k + 1}"] = float(vals[order_neg[k]])
            rows.append(row)
        local = pd.DataFrame(rows)
        local.to_csv(
            paths["models_dir"] / "shap_top_risk_customer_reasons.csv", index=False
        )

        try:
            top = global_importance.head(20).sort_values(
                "mean_abs_shap", ascending=True
            )
            plt.figure(figsize=(8, 6))
            plt.barh(top["feature"], top["mean_abs_shap"])
            plt.title("Global SHAP Importance for Churn Model")
            plt.xlabel("Mean absolute SHAP value")
            plt.tight_layout()
            plt.savefig(
                paths["visualization_exports_dir"] / "m5_shap_global_importance.png",
                dpi=160,
            )
            plt.close()
        except Exception as plot_exc:
            print(f"[M5] SHAP plot skipped: {plot_exc}")
        status_path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "sample_size": int(n_sample),
                    "top_risk_customers": int(len(rows)),
                    "note": "SHAP values explain model prediction associations, not causality.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return global_importance
    except Exception as exc:
        status_path.write_text(
            json.dumps({"status": "failed", "reason": str(exc)}, indent=2),
            encoding="utf-8",
        )
        print(f"[M5] SHAP export failed: {exc}")
        return pd.DataFrame()


def create_seasonality_audit(
    paths: Dict[str, Path], id_col: str, cut_off_day: int, horizon: int
) -> pd.DataFrame:
    txn_path = paths["transaction_master_parquet"]
    txns = pd.read_parquet(txn_path, columns=[id_col, "DAY", "SALES_VALUE"])
    txns[id_col] = txns[id_col].astype(int)
    txns["week_index"] = (txns["DAY"] // 7).astype(int)
    weekly = txns.groupby("week_index", as_index=False).agg(
        total_sales=("SALES_VALUE", "sum"),
        txn_count=("DAY", "size"),
        active_households=(id_col, "nunique"),
        first_day=("DAY", "min"),
        last_day=("DAY", "max"),
    )
    weekly["period"] = np.where(
        weekly["last_day"] < cut_off_day,
        "observation",
        np.where(
            weekly["first_day"] > cut_off_day + horizon,
            "after_horizon",
            "prediction_window",
        ),
    )
    weekly.to_csv(paths["models_dir"] / "seasonality_audit.csv", index=False)

    def window_stats(name: str, start: int, end: int) -> Dict[str, Any]:
        w = txns[(txns["DAY"] >= start) & (txns["DAY"] <= end)]
        return {
            "window": name,
            "start_day": start,
            "end_day": end,
            "total_sales": float(w["SALES_VALUE"].sum()),
            "txn_count": int(len(w)),
            "active_households": int(w[id_col].nunique()),
            "avg_sales_per_active_household": float(
                w["SALES_VALUE"].sum() / max(w[id_col].nunique(), 1)
            ),
        }

    windows = [
        window_stats(
            "pre_cutoff_recent_60d", max(0, cut_off_day - horizon), cut_off_day - 1
        ),
        window_stats(
            "pre_cutoff_previous_60d",
            max(0, cut_off_day - 2 * horizon),
            max(0, cut_off_day - horizon - 1),
        ),
        window_stats("prediction_window_60d", cut_off_day, cut_off_day + horizon),
    ]
    win_df = pd.DataFrame(windows)
    base_sales = float(
        win_df.loc[win_df["window"] == "pre_cutoff_recent_60d", "total_sales"].iloc[0]
    )
    win_df["sales_index_vs_recent_pre_cutoff"] = win_df["total_sales"] / max(
        base_sales, 1e-9
    )
    win_df.to_csv(
        paths["models_dir"] / "seasonality_window_comparison.csv", index=False
    )

    plt.figure(figsize=(10, 5))
    plt.plot(weekly["week_index"], weekly["total_sales"], marker="o", markersize=3)
    plt.axvline(cut_off_day // 7, linestyle="--", label="cut-off week")
    plt.xlabel("Week index")
    plt.ylabel("Total weekly sales")
    plt.title("Weekly Revenue Trend and Prediction Window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        paths["visualization_exports_dir"] / "m5_weekly_revenue_trend.png", dpi=160
    )
    plt.close()
    return weekly


def save_visual_exports(importance: pd.DataFrame, paths: Dict[str, Path]) -> None:
    if not importance.empty:
        top = importance.head(15).sort_values("importance", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["importance"])
        plt.title("Native Churn Model Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(
            paths["visualization_exports_dir"] / "m5_feature_importance_top15.png",
            dpi=160,
        )
        plt.close()
    cal_path = paths["models_dir"] / "calibration_by_decile.csv"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        test = cal[cal["dataset"] == "test"]
        if not test.empty:
            plt.figure(figsize=(6, 5))
            plt.plot(
                test["mean_predicted_probability"],
                test["actual_churn_rate"],
                marker="o",
                label="test deciles",
            )
            hi = max(
                test["mean_predicted_probability"].max(),
                test["actual_churn_rate"].max(),
                0.01,
            )
            plt.plot([0, hi], [0, hi], linestyle="--", label="perfect calibration")
            plt.xlabel("Mean predicted churn probability")
            plt.ylabel("Actual churn rate")
            plt.title("M5 Calibration Curve by Decile")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                paths["visualization_exports_dir"] / "m5_calibration_curve.png", dpi=160
            )
            plt.close()


def write_resampling_audit(paths: Dict[str, Path], run_smote: bool) -> None:
    text = f"""# M5 Imbalance and Resampling Audit

## Rule
Validation and test sets are never resampled. Test data must preserve the real churn distribution.

## Current pipeline
- Train/validation/test split is created before any optional resampling experiment.
- Main churn models use class weighting and probability calibration, not SMOTE, because business formulas require meaningful probabilities.
- Optional SMOTETomek baseline enabled: `{run_smote}`.
- If enabled, SMOTETomek is wrapped inside an `imblearn.pipeline.Pipeline`, so resampling happens only inside the training pipeline.

## Why this matters
SMOTE before train/test split would leak synthetic information into validation/test and inflate model quality. M5 therefore treats SMOTE as an optional benchmark, never as a pre-processing artifact from M4.
"""
    (paths["reports_internal_dir"] / "M5_imbalance_and_resampling_audit.md").write_text(
        text, encoding="utf-8"
    )


def write_report_outline(
    metrics: pd.DataFrame,
    cal: pd.DataFrame,
    active_metrics: pd.DataFrame,
    value_metrics: pd.DataFrame,
    shap_df: pd.DataFrame,
    predictions: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    best = metrics[metrics["model"] != "Dummy prior"].iloc[0]
    selected = cal.iloc[0]
    active = active_metrics.iloc[0]
    value = value_metrics.iloc[0]
    base_profitable = int(
        predictions.get("profitable_to_treat_base", pd.Series(dtype=bool)).sum()
    )
    shap_top = (
        shap_df.head(5)["feature"].tolist()
        if shap_df is not None and not shap_df.empty
        else []
    )
    md = f"""# M5 Modeling Report Outline — v3

## Role
M5 receives M4's delivered feature table and builds a risk/value/profit ranking for M6. M5 does not claim treatment causality; M6 validates causal lift through A/B testing.

## Methodology
- Input feature table: `models/final_ML_features.csv` from M4.
- `household_key` is kept only as an identifier and excluded from training.
- Churn is interpreted as an operational retail churn label, not a formal cancellation event.
- Churn models are benchmarked with PR-AUC and F2-score because the churn class is imbalanced.
- The selected churn model is calibrated before probability is used in business formulas.
- Value modeling is upgraded from a single conditional model to a two-part discounted value model:
  1. `p_future_active`: probability that a customer generates future revenue in the prediction window.
  2. `predicted_discounted_value_60d_if_active`: discounted 60-day value conditional on being active.
  3. `predicted_expected_discounted_value_60d = p_future_active × value_if_active`.
- Expected incremental profit uses discounted value and scenario assumptions.
- Model selection prioritizes calibrated probability quality for business formulas, not only raw ranking performance.
- XGBoost is disabled by default for runtime/reproducibility; it can be enabled in `config/paths.yaml` for additional benchmarking.
- SHAP outputs are used for model explainability, but they are predictive explanations, not causal claims.

## Churn model
- Selected reporting model: **{best["model"]}**
- Calibration selected: **{selected["calibration_method"]}**
- Threshold on calibrated probability: **{selected["val_threshold"]:.2f}**
- Test PR-AUC: **{selected["test_PR_AUC"]:.4f}**
- Test F2-score: **{selected["test_F2_score"]:.4f}**
- Test Brier score: **{selected["test_brier_score"]:.4f}**
- Test mean predicted probability: **{selected["test_mean_predicted_probability"]:.4f}** vs actual churn rate **{selected["test_actual_positive_rate"]:.4f}**

The selected model should be interpreted as the reporting/champion model for calibrated decision support, not as proof that it is statistically superior to every alternative. With a small sample size and unstable positive-class counts, differences across candidate models may fall within split-level variance.

## Two-part discounted value model
- Future-active model: **{active["active_model"]} ({active["calibration_method"]})**
- Active-model test Brier score: **{active["test_brier_score"]:.4f}**
- Conditional value model: **{value["value_model"]}**
- Conditional value target: **log1p(discounted_future_revenue_60d | future_active=1)**
- Test RMSE_log: **{value["test_RMSE_log"]:.4f}**

## Expected incremental profit and candidate ranking
`Expected Incremental Profit_i = p_churn_calibrated_i × save_rate_given_treatment × predicted_discounted_value_60d_if_active_i × gross_margin - treatment_cost_i`

The profit formula intentionally uses `predicted_discounted_value_60d_if_active`, not the unconditional `p_future_active × value_if_active`, to avoid double-counting the active/churn probability. Under the base scenario, **{base_profitable}** customers have positive expected incremental treatment profit.

If no customers have positive expected profit under the base scenario, `priority_rank` falls back to churn-risk ranking for A/B test candidate selection. In that case, profit ranking should not be interpreted as treatment eligibility; it is only a scenario diagnostic.

## SHAP top drivers
{chr(10).join([f"- {x}" for x in shap_top]) if shap_top else "- SHAP was skipped or unavailable; use `models/feature_importance.csv` as fallback."}

## Important limitations
- M5 uses M4's current feature table as-is. A separate M4 feature lineage audit is still recommended.
- Discounted 60-day value is not true lifetime CLV.
- SHAP and feature importance are associations, not causal treatment effects.
- Seasonality is audited, but not fully modeled as a time-series forecasting problem.
"""
    (paths["reports_internal_dir"] / "M5_modeling_report_outline.md").write_text(
        md, encoding="utf-8"
    )


def update_model_readme(paths: Dict[str, Path], run_smote: bool) -> None:
    readme = f"""# README — M5 Model Pipeline v3

This README explains the M5 pipeline for a beginner. M5 builds churn-risk, discounted value, and expected-profit rankings for the A/B testing step.

## 1. What M5 outputs

For each household, M5 outputs:

| Output | Meaning |
|---|---|
| `p_churn_calibrated` | Calibrated churn-risk probability under the project churn definition |
| `p_future_active` | Probability that the household will generate revenue in the 60-day prediction window |
| `predicted_discounted_value_60d_if_active` | Discounted 60-day revenue if the household is active |
| `predicted_expected_discounted_value_60d` | Two-part unconditional expected value = `p_future_active × value_if_active`; useful as a value diagnostic, not the direct profit input |
| `expected_incremental_profit_*` | Scenario-based incremental profit from targeting, calculated with `value_if_active` to avoid double-counting activity probability |
| `priority_segment` | Risk/value segment for business interpretation |

M5 does **not** prove that a voucher works. M6 must validate causal treatment lift with A/B testing.

## 2. Algorithms used

### Churn classification
The pipeline benchmarks:

- Dummy prior baseline
- Logistic Regression with class balancing
- Random Forest with class balancing
- Extra Trees with class balancing
- Optional XGBoost weighted model if `modeling.run_xgboost: true`
- Optional SMOTETomek baseline if `modeling.run_smote_baseline: true`

Current config uses `cv_folds: 5` and `tuning_n_iter: 10` for the Logistic Regression parameter search. Tree models can be enabled as lightweight comparison baselines with `modeling.run_tree_baselines: true`; the default GitHub-friendly run keeps them disabled for runtime reproducibility. This is still not exhaustive optimization. XGBoost is disabled by default for reproducibility/runtime, not because it is invalid. Enable `modeling.run_xgboost: true` to benchmark it under the same pipeline.

Success metrics:

| Metric | Meaning |
|---|---|
| PR-AUC | Ranking quality for rare churn cases |
| F2-score | Churn classification metric that prioritizes recall over precision |
| Recall | Share of actual churners caught |
| Precision | Share of predicted churn-risk customers who actually churn |
| Brier score | Probability quality/calibration |

### Probability calibration
The champion churn model is calibrated using validation data. The pipeline compares raw, sigmoid, and isotonic probabilities, then selects the best validation Brier score. Business formulas use `p_churn_calibrated`, not raw weighted-model probability. Isotonic calibration may create stepwise/flat probability groups; this is acceptable for calibration quality but should not be overinterpreted as a perfectly continuous risk score.

### Two-part discounted value model
M5 no longer treats value as a single simple CLV model. It uses:

1. Active model: predicts `p_future_active`.
2. Conditional value model: predicts `discounted_future_revenue_60d` only among future-active customers.
3. Combined expected value: `p_future_active × predicted_discounted_value_60d_if_active`.

The discount rate is configured in `config/paths.yaml`. A 60-day discounted value is more precise than calling it full lifetime CLV.

### Expected incremental profit

```text
p_churn_calibrated × save_rate_given_treatment × predicted_discounted_value_60d_if_active × gross_margin - treatment_cost
```

This is scenario-based. The save rate, margin, and treatment cost are assumptions until M6 validates lift with A/B testing. The formula uses `predicted_discounted_value_60d_if_active`, not unconditional expected value, to avoid multiplying by both `p_churn` and `p_future_active`. If no customers have positive expected profit under the base scenario, M5 uses churn-risk ranking for A/B test candidate selection rather than profit ranking for auto-targeting. A base scenario with zero profitable customers is a business-economics finding: it suggests that treatment cost, save rate, margin, or expected value assumptions must improve before rollout.

## 3. Data leakage and SMOTE rules

- M5 assumes M4's delivered features are computed before `cut_off_day`.
- M5 does not modify M4 features.
- Validation and test sets are never resampled.
- Optional SMOTETomek enabled in this run: `{run_smote}`.
- If SMOTETomek is enabled, it is inside an imbalanced-learn Pipeline after the split, not applied to the full dataset.

## 4. SHAP explainability

M5 exports SHAP outputs when the `shap` package is available:

| File | Meaning |
|---|---|
| `models/shap_global_importance.csv` | Global mean absolute SHAP drivers |
| `models/shap_top_risk_customer_reasons.csv` | Local explanations for top-risk customers |
| `visualization/exports/m5_shap_global_importance.png` | Report-ready SHAP importance chart |

Important: SHAP explains model predictions. It does not prove causal treatment effects.

## 5. Seasonality audit

M5 adds a lightweight seasonality check:

| File | Meaning |
|---|---|
| `models/seasonality_audit.csv` | Weekly sales, transactions, and active households |
| `models/seasonality_window_comparison.csv` | Prediction-window revenue compared with recent pre-cutoff windows |
| `visualization/exports/m5_weekly_revenue_trend.png` | Weekly sales trend with cut-off marker |

This does not fully model seasonality, but it documents whether the 60-day prediction window looks unusual.

## 6. How files interact

```text
config/paths.yaml
  -> scripts/utils.py resolves paths
  -> scripts/modeling.py runs the full pipeline
  -> scripts/evaluation.py provides metrics/helpers
  -> models/final_ML_features.csv is the M4 handoff
  -> Data/Processed/transactions_master.parquet builds discounted value labels and seasonality audit
  -> models/*.csv contains outputs for M5/M6
```

## 7. GitHub/data governance note

Raw/processed/intermediate Dunnhumby data and customer-level generated outputs are not intended to be pushed to a public GitHub repository. See `Data/README.md` for local data placement. The repository should track code, configuration, lightweight metric summaries, and documentation; customer-level predictions and heavy artifacts should be regenerated locally.

## 8. How to run

```bash
pip install -r requirements.txt
python -m py_compile scripts/*.py
python scripts/modeling.py --config config/paths.yaml
```

## 9. Main output files

- `models/model_metrics.csv`
- `models/calibration_summary.csv`
- `models/active_model_metrics.csv`
- `models/value_model_metrics.csv`
- `models/discounted_value_labels.csv`
- `models/churn_predictions.csv`
- `models/high_risk_customers_for_ab_test.csv`
- `models/scenario_profit_summary.csv`
- `models/scenario_sensitivity_grid.csv`
- `models/break_even_analysis.csv`
- `models/active_churn_target_overlap_audit.csv`
- `models/shap_global_importance.csv`
- `models/shap_top_risk_customer_reasons.csv`
- `models/seasonality_audit.csv`
- `models/seasonality_window_comparison.csv`
- `models/model.pkl`

## 10. What to say in the report

Use wording like:

> We estimate calibrated churn risk and two-part discounted 60-day customer value, then combine them with scenario-based save rate, gross margin, and treatment cost to prioritize customers for A/B testing.

Avoid saying:

- “This is true lifetime CLV.”
- “SHAP proves why the customer churned.”
- “High-risk customers should automatically receive vouchers.”
"""
    (paths["project_root"] / "README_M5_MODEL.md").write_text(readme, encoding="utf-8")


def relocate_internal_diagnostic_outputs(paths: Dict[str, Path]) -> None:
    """Move non-core diagnostic CSVs from models/ to reports/internal_briefs/.

    Core model outputs remain in models/. Diagnostic/audit files are still generated by
    the M5 pipeline, but they are report/internal review artifacts rather than model
    scoring artifacts.
    """
    diagnostic_files = [
        "feature_lineage_audit_template.csv",
        "multicollinearity_vif.csv",
        "numeric_feature_correlation_pairs.csv",
        "value_model_residual_summary.csv",
        "value_model_decile_diagnostics.csv",
        "voucher_diversification_audit.csv",
        "recommendation_merge_audit.csv",
        "split_stability_runs.csv",
        "split_stability_summary.csv",
        "seasonality_audit.csv",
        "seasonality_window_comparison.csv",
    ]
    for name in diagnostic_files:
        src = paths["models_dir"] / name
        dst = paths["reports_internal_dir"] / name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()
            src.rename(dst)


def run_m5_pipeline(config_path: str | Path | None = None) -> Dict[str, Any]:
    project_root = find_project_root()
    config = load_config(config_path=config_path, project_root=project_root)
    paths = resolve_project_paths(config, project_root)
    ensure_project_structure(paths)
    cfg = config["modeling"]
    id_col = cfg["id_col"]
    target_col = cfg["target_col"]
    categorical_cols = list(cfg.get("categorical_cols", []))
    cut_off_day = int(cfg["cut_off_day"])
    random_state = int(cfg["random_state"])
    f_beta = float(cfg.get("f_beta", 2))
    n_estimators = int(cfg.get("n_estimators", 50))
    n_jobs = int(cfg.get("n_jobs", 1))
    test_size = float(cfg.get("test_size", 0.20))
    validation_size = float(cfg.get("validation_size", 0.25))
    cv_folds = int(cfg.get("cv_folds", 2))
    tuning_n_iter = int(cfg.get("tuning_n_iter", 2))
    run_xgboost = bool(cfg.get("run_xgboost", False))
    run_tree_baselines = bool(cfg.get("run_tree_baselines", True))
    run_tree_value = bool(cfg.get("run_tree_value_models", False))
    run_smote = bool(cfg.get("run_smote_baseline", False))
    value_cfg = config.get("value_modeling", {})
    prediction_horizon_days = int(value_cfg.get("prediction_horizon_days", 60))
    annual_discount_rate = float(value_cfg.get("default_annual_discount_rate", 0.08))
    shap_cfg = config.get("explainability", {})
    run_shap = bool(shap_cfg.get("run_shap", True))
    shap_sample_size = int(shap_cfg.get("shap_sample_size", 500))
    shap_top_n = int(shap_cfg.get("shap_top_n_customers", 30))

    print("[M5] Project root:", project_root)
    features = load_feature_table(
        paths["feature_table_csv"], id_col, target_col, categorical_cols
    )
    audit_inputs(features, paths, id_col, target_col, cut_off_day)
    features.to_parquet(paths["models_dir"] / "final_features.parquet", index=False)
    feature_cols = [c for c in features.columns if c not in [id_col, target_col]]
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    split = prepare_splits(
        features,
        id_col,
        target_col,
        feature_cols,
        test_size,
        validation_size,
        random_state,
    )
    linear_pre, tree_pre = build_preprocessors(num_cols, cat_cols)
    scale_pos_weight = float(
        (split["y_train"] == 0).sum() / max((split["y_train"] == 1).sum(), 1)
    )

    specs = build_classifier_specs(
        linear_pre,
        tree_pre,
        feature_cols,
        scale_pos_weight,
        random_state,
        n_jobs,
        run_xgboost,
        run_tree_baselines,
    )
    smote = build_smote_baseline(num_cols, random_state) if run_smote else None
    metrics, _tuning, fitted = tune_and_benchmark_classifiers(
        specs,
        split,
        f_beta,
        paths["models_dir"],
        random_state,
        cv_folds,
        tuning_n_iter,
        smote,
    )
    (
        champion_name,
        calibration_method,
        threshold,
        churn_model,
        churn_cols,
        cal_summary,
    ) = select_calibrated_churn_model(
        metrics, fitted, split, paths["models_dir"], f_beta
    )
    importance = export_feature_importance(churn_model, churn_cols, paths["models_dir"])

    value_labels = build_discounted_value_labels(
        features,
        paths,
        id_col,
        cut_off_day,
        prediction_horizon_days,
        annual_discount_rate,
    )
    (
        active_metrics,
        value_metrics,
        active_model,
        active_model_name,
        value_model,
        value_model_name,
    ) = train_two_part_value_model(
        value_labels,
        split,
        feature_cols,
        linear_pre,
        tree_pre,
        paths["models_dir"],
        id_col,
        random_state,
        f_beta,
        n_estimators,
        n_jobs,
        run_tree_value,
        run_xgboost,
    )
    predictions = score_customers(
        features,
        value_labels,
        churn_model,
        churn_cols,
        threshold,
        active_model,
        value_model,
        value_model_name,
        feature_cols,
        id_col,
        target_col,
        config["expected_profit_scenarios"],
        paths,
        cut_off_day,
        annual_discount_rate,
        active_model_name,
        champion_name,
        calibration_method,
        top_k_shares=config.get("evaluation", {}).get(
            "top_k_shares", [0.05, 0.10, 0.20]
        ),
        decile_count=int(config.get("evaluation", {}).get("decile_count", 10)),
        recommendation_top_n=int(
            config.get("recommendations", {}).get("top_n_vouchers", 3)
        ),
    )

    if bool(config.get("seasonality_audit", {}).get("run", True)):
        create_seasonality_audit(paths, id_col, cut_off_day, prediction_horizon_days)
    shap_df = (
        export_shap_outputs(
            churn_model,
            churn_cols,
            features,
            predictions,
            paths,
            shap_sample_size,
            shap_top_n,
        )
        if run_shap
        else pd.DataFrame()
    )
    save_visual_exports(importance, paths)
    write_resampling_audit(paths, run_smote)
    write_report_outline(
        metrics, cal_summary, active_metrics, value_metrics, shap_df, predictions, paths
    )
    update_model_readme(paths, run_smote)
    relocate_internal_diagnostic_outputs(paths)

    selected = cal_summary.iloc[0]
    summary = {
        "version": "v3_discounted_two_part_value_shap_seasonality",
        "project_root": str(project_root),
        "feature_rows": int(len(features)),
        "churn_rate": float(features[target_col].mean()),
        "champion_churn_model": champion_name,
        "calibration_method": calibration_method,
        "champion_threshold": float(threshold),
        "test_PR_AUC_calibrated": float(selected["test_PR_AUC"]),
        "test_F2_score_calibrated": float(selected["test_F2_score"]),
        "test_brier_score_calibrated": float(selected["test_brier_score"]),
        "active_model": active_model_name,
        "value_model": value_model_name,
        "annual_discount_rate": annual_discount_rate,
        "base_profitable_customers": int(
            predictions.get("profitable_to_treat_base", pd.Series(dtype=bool)).sum()
        ),
        "shap_status_file": str(paths["reports_internal_dir"] / "M5_shap_status.json"),
        "outputs_dir": str(paths["models_dir"]),
    }
    (paths["reports_internal_dir"] / "m5_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[M5] Pipeline complete.")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M5 v3 churn, discounted value, SHAP, and seasonality pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/paths.yaml",
        help="Path to YAML config file, relative to project root by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_m5_pipeline(args.config)
