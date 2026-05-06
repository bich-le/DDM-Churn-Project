\
"""End-to-end M5 modeling pipeline: churn, 60-day value, and expected profit.

Run from the project root:
    python scripts/modeling.py --config config/paths.yaml

Main outputs:
    models/model_metrics.csv
    models/champion_test_metrics.csv
    models/clv_model_metrics.csv
    models/feature_importance.csv
    models/churn_predictions.csv
    models/high_risk_customers_for_ab_test.csv
    models/profitable_treatment_candidates_base.csv
    models/model.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")

# Allow `python scripts/modeling.py` from project root without package installation.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.evaluation import (
    best_fbeta_threshold,
    evaluate_classifier,
    extract_feature_importance,
    make_one_hot_encoder,
    regression_metrics,
)
from scripts.utils import ensure_project_structure, find_project_root, load_config, resolve_project_paths

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = str(exc)

try:
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    IMBLEARN_AVAILABLE = False
    SMOTETomek = None
    ImbPipeline = None
    IMBLEARN_IMPORT_ERROR = str(exc)


def load_feature_table(feature_path: Path, id_col: str, target_col: str, categorical_cols: List[str]) -> pd.DataFrame:
    """Load and lightly validate the customer-level modeling table."""
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")

    features = pd.read_csv(feature_path)
    required = {id_col, target_col}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table missing required columns: {sorted(missing)}")

    features[id_col] = features[id_col].astype(int)
    for col in categorical_cols:
        if col in features.columns:
            features[col] = features[col].astype(str)
    features = features.drop_duplicates(id_col).copy()
    return features


def audit_inputs(
    features: pd.DataFrame,
    paths: Dict[str, Path],
    id_col: str,
    target_col: str,
    cut_off_day: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create audit files and investigate any customer-base rows missing from features."""
    audit_rows: List[Tuple[str, Any]] = [
        ("feature_rows", len(features)),
        ("feature_columns", features.shape[1]),
        ("duplicate_household_key", int(features[id_col].duplicated().sum())),
        ("missing_values_total", int(features.isna().sum().sum())),
        ("churn_count", int(features[target_col].sum())),
        ("non_churn_count", int((features[target_col] == 0).sum())),
        ("churn_rate", float(features[target_col].mean())),
    ]

    missing_investigation = pd.DataFrame()
    customer_path = paths["customer_base_parquet"]
    txn_path = paths["transaction_master_parquet"]

    if customer_path.exists():
        customers = pd.read_parquet(customer_path)
        customers[id_col] = customers[id_col].astype(int)
        missing_households = sorted(set(customers[id_col]) - set(features[id_col]))
        audit_rows += [
            ("customer_base_rows", len(customers)),
            ("missing_from_features_count", len(missing_households)),
            ("missing_from_features_households", ", ".join(map(str, missing_households[:50]))),
        ]

        if missing_households and txn_path.exists():
            txns = pd.read_parquet(txn_path, columns=[id_col, "DAY", "SALES_VALUE"])
            txns[id_col] = txns[id_col].astype(int)
            miss_txns = txns[txns[id_col].isin(missing_households)].copy()
            if not miss_txns.empty:
                missing_investigation = (
                    miss_txns.groupby(id_col)
                    .agg(
                        txn_rows=("DAY", "size"),
                        first_day=("DAY", "min"),
                        last_day=("DAY", "max"),
                        pre_cutoff_txn_rows=("DAY", lambda s: int((s < cut_off_day).sum())),
                        post_cutoff_txn_rows=("DAY", lambda s: int((s >= cut_off_day).sum())),
                        total_sales=("SALES_VALUE", "sum"),
                    )
                    .reset_index()
                )
    else:
        audit_rows.append(("customer_base_check_error", f"Missing file: {customer_path}"))

    audit_df = pd.DataFrame(audit_rows, columns=["check", "value"])
    audit_df.to_csv(paths["reports_internal_dir"] / "m5_data_audit.csv", index=False)
    if not missing_investigation.empty:
        missing_investigation.to_csv(paths["intermediate_analysis_dir"] / "m5_missing_household_investigation.csv", index=False)
    return audit_df, missing_investigation


def prepare_splits(
    features: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_cols: List[str],
    test_size: float,
    validation_size: float,
    random_state: int,
) -> Dict[str, Any]:
    """Create stratified train/validation/test splits."""
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


def build_preprocessors(num_cols: List[str], cat_cols: List[str]) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """Create preprocessing transformers for linear and tree models."""
    scale_ohe_preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_one_hot_encoder(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    tree_preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", make_one_hot_encoder(), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return scale_ohe_preprocess, tree_preprocess


def build_classifier_specs(
    scale_ohe_preprocess: ColumnTransformer,
    tree_preprocess: ColumnTransformer,
    feature_cols: List[str],
    num_cols: List[str],
    scale_pos_weight: float,
    random_state: int,
    n_estimators: int,
    n_jobs: int,
) -> List[Tuple[str, Any, List[str]]]:
    """Build benchmark classifier specifications."""
    model_specs: List[Tuple[str, Any, List[str]]] = []

    model_specs.append((
        "Dummy prior",
        Pipeline([("model", DummyClassifier(strategy="prior", random_state=random_state))]),
        feature_cols,
    ))
    model_specs.append((
        "Logistic Regression balanced",
        Pipeline([
            ("preprocess", scale_ohe_preprocess),
            ("model", LogisticRegression(class_weight="balanced", max_iter=3000, solver="liblinear", random_state=random_state)),
        ]),
        feature_cols,
    ))
    model_specs.append((
        "Random Forest balanced",
        Pipeline([
            ("preprocess", tree_preprocess),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=n_jobs,
            )),
        ]),
        feature_cols,
    ))
    model_specs.append((
        "Extra Trees balanced",
        Pipeline([
            ("preprocess", tree_preprocess),
            ("model", ExtraTreesClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=n_jobs,
            )),
        ]),
        feature_cols,
    ))

    if XGBOOST_AVAILABLE:
        model_specs.append((
            "XGBoost weighted",
            Pipeline([
                ("preprocess", tree_preprocess),
                ("model", XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    scale_pos_weight=scale_pos_weight,
                    reg_lambda=5.0,
                    min_child_weight=3,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )),
            ]),
            feature_cols,
        ))

    if IMBLEARN_AVAILABLE:
        model_specs.append((
            "SMOTETomek + Logistic Regression",
            ImbPipeline([
                ("scale", StandardScaler()),
                ("resample", SMOTETomek(random_state=random_state)),
                ("model", LogisticRegression(max_iter=3000, solver="liblinear", random_state=random_state)),
            ]),
            num_cols,
        ))

    return model_specs


def benchmark_classifiers(
    model_specs: List[Tuple[str, Any, List[str]]],
    split: Dict[str, Any],
    f_beta: float,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[Any, List[str]]]]:
    """Fit classifier benchmarks and save metrics."""
    results: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Tuple[Any, List[str]]] = {}

    for name, model, cols in model_specs:
        print(f"[M5] Fitting classifier: {name}")
        model.fit(split["X_train"][cols], split["y_train"])
        fitted_models[name] = (model, cols)

        val_proba = model.predict_proba(split["X_val"][cols])[:, 1]
        threshold, _ = best_fbeta_threshold(split["y_val"], val_proba, beta=f_beta)
        val_metrics = evaluate_classifier(model, split["X_val"][cols], split["y_val"], threshold, beta=f_beta)
        test_metrics = evaluate_classifier(model, split["X_test"][cols], split["y_test"], threshold, beta=f_beta)

        row: Dict[str, Any] = {"model": name, "features_used": "numeric+categorical" if cols == list(split["X"].columns) else "numeric_only"}
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value
        for key, value in test_metrics.items():
            row[f"test_{key}"] = value
        results.append(row)

    metrics_df = pd.DataFrame(results).sort_values(["val_PR_AUC", "val_F2_score"], ascending=False).reset_index(drop=True)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    return metrics_df, fitted_models


def finalize_champion(
    metrics_df: pd.DataFrame,
    fitted_models: Dict[str, Tuple[Any, List[str]]],
    split: Dict[str, Any],
    feature_cols: List[str],
    output_dir: Path,
    f_beta: float,
) -> Tuple[str, float, Any, List[str], pd.DataFrame]:
    """Select champion model, evaluate final holdout, and export feature importance."""
    champion_name = str(metrics_df.iloc[0]["model"])
    champion_threshold = float(metrics_df.iloc[0]["val_threshold"])
    champion_template, champion_cols = fitted_models[champion_name]

    final_test_metrics = evaluate_classifier(
        champion_template,
        split["X_test"][champion_cols],
        split["y_test"],
        champion_threshold,
        beta=f_beta,
    )
    final_test_df = pd.DataFrame([{f"final_test_{k}": v for k, v in final_test_metrics.items()}])
    final_test_df.insert(0, "champion_model", champion_name)
    final_test_df.to_csv(output_dir / "champion_test_metrics.csv", index=False)

    importance_df = extract_feature_importance(champion_template, champion_cols)
    if not importance_df.empty:
        importance_df["business_interpretation"] = importance_df["feature"].map(feature_business_interpretation)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    return champion_name, champion_threshold, champion_template, champion_cols, importance_df


def feature_business_interpretation(feature: str) -> str:
    """Brief interpretation for common churn features."""
    f = feature.lower()
    if "recency" in f or "inactive" in f or "days_since" in f:
        return "Recent inactivity or time since engagement; usually linked with churn risk."
    if "frequency" in f or "freq" in f:
        return "Purchase frequency / activity intensity."
    if "monetary" in f or "basket" in f or "sales" in f or "clv" in f:
        return "Customer spending level or basket value."
    if "promo" in f or "coupon" in f or "campaign" in f or "mailer" in f or "display" in f:
        return "Promotion, coupon, or campaign responsiveness."
    if "brand" in f:
        return "Brand preference pattern, such as private-brand reliance."
    if "slope" in f or "trend" in f:
        return "Behavioral trend over time."
    if "store" in f:
        return "Primary store/location pattern encoded as categorical information."
    if "demographic" in f:
        return "Availability of demographic profile information."
    return "Predictive feature used by the model."


def build_clv_labels(features: pd.DataFrame, paths: Dict[str, Path], id_col: str, cut_off_day: int) -> pd.DataFrame:
    """Create 60-day future revenue labels from post-cutoff transactions."""
    txn_path = paths["transaction_master_parquet"]
    if not txn_path.exists():
        raise FileNotFoundError(f"Transaction file not found for CLV labels: {txn_path}")
    txns = pd.read_parquet(txn_path, columns=[id_col, "DAY", "SALES_VALUE"])
    txns[id_col] = txns[id_col].astype(int)
    future = txns[txns["DAY"] >= cut_off_day].copy()
    future_revenue = future.groupby(id_col, as_index=False)["SALES_VALUE"].sum()
    future_revenue = future_revenue.rename(columns={"SALES_VALUE": "future_revenue_60d"})
    clv_data = features.merge(future_revenue, on=id_col, how="left")
    clv_data["future_revenue_60d"] = clv_data["future_revenue_60d"].fillna(0).astype(float)
    clv_data["future_active_flag"] = (clv_data["future_revenue_60d"] > 0).astype(int)
    clv_data.to_csv(paths["models_dir"] / "clv_training_labels.csv", index=False)
    return clv_data


def build_regressor_specs(
    scale_ohe_preprocess: ColumnTransformer,
    tree_preprocess: ColumnTransformer,
    random_state: int,
    n_estimators: int,
    n_jobs: int,
) -> List[Tuple[str, Any]]:
    """Build CLV conditional-value regression candidates."""
    reg_specs: List[Tuple[str, Any]] = [
        ("Ridge Regression", Pipeline([("preprocess", scale_ohe_preprocess), ("model", Ridge(alpha=1.0))])),
        ("Random Forest Regressor", Pipeline([
            ("preprocess", tree_preprocess),
            ("model", RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=n_jobs,
            )),
        ])),
    ]
    if XGBOOST_AVAILABLE:
        reg_specs.append((
            "XGBoost Regressor",
            Pipeline([
                ("preprocess", tree_preprocess),
                ("model", XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="reg:squarederror",
                    reg_lambda=5.0,
                    min_child_weight=3,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )),
            ]),
        ))
    return reg_specs


def benchmark_clv_models(
    clv_data: pd.DataFrame,
    split: Dict[str, Any],
    feature_cols: List[str],
    reg_specs: List[Tuple[str, Any]],
    output_dir: Path,
    id_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], str, Any]:
    """Train CLV regressors on future-active customers and select best by validation RMSE."""
    id_sets = {name: set(split[f"ids_{name}"].astype(int)) for name in ["train", "val", "test"]}
    clv_train = clv_data[clv_data[id_col].isin(id_sets["train"])].copy()
    clv_val = clv_data[clv_data[id_col].isin(id_sets["val"])].copy()
    clv_test = clv_data[clv_data[id_col].isin(id_sets["test"])].copy()

    clv_train_pos = clv_train[clv_train["future_revenue_60d"] > 0].copy()
    clv_val_pos = clv_val[clv_val["future_revenue_60d"] > 0].copy()
    clv_test_pos = clv_test[clv_test["future_revenue_60d"] > 0].copy()

    if clv_train_pos.empty:
        raise ValueError("No positive future revenue customers in training split; cannot train conditional CLV model.")

    y_train = np.log1p(clv_train_pos["future_revenue_60d"])
    y_val = np.log1p(clv_val_pos["future_revenue_60d"])
    y_test = np.log1p(clv_test_pos["future_revenue_60d"])

    clv_results: List[Dict[str, Any]] = []
    clv_models: Dict[str, Any] = {}
    for name, reg in reg_specs:
        print(f"[M5] Fitting CLV model: {name}")
        reg.fit(clv_train_pos[feature_cols], y_train)
        clv_models[name] = reg
        val_pred = reg.predict(clv_val_pos[feature_cols]) if len(clv_val_pos) else np.array([])
        test_pred = reg.predict(clv_test_pos[feature_cols]) if len(clv_test_pos) else np.array([])
        row: Dict[str, Any] = {"clv_model": name}
        row.update(regression_metrics(y_val, val_pred, "val"))
        row.update(regression_metrics(y_test, test_pred, "test"))
        clv_results.append(row)

    clv_metrics_df = pd.DataFrame(clv_results).sort_values("val_RMSE_log", ascending=True).reset_index(drop=True)
    clv_metrics_df.to_csv(output_dir / "clv_model_metrics.csv", index=False)
    clv_champion_name = str(clv_metrics_df.iloc[0]["clv_model"])
    clv_champion = clv_models[clv_champion_name]
    return clv_metrics_df, clv_models, clv_champion_name, clv_champion


def calculate_deciles_and_segments(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add risk/value deciles and human-readable priority segments."""
    predictions = predictions.copy()
    predictions["risk_decile"] = pd.qcut(
        predictions["p_churn"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)
    predictions["value_decile"] = pd.qcut(
        predictions["predicted_CLV_60d_if_retained"].rank(method="first", ascending=False),
        q=10,
        labels=list(range(1, 11)),
    ).astype(int)
    risk_high = predictions["risk_decile"] <= 3
    value_high = predictions["value_decile"] <= 3
    predictions["priority_segment"] = np.select(
        [risk_high & value_high, risk_high & (~value_high), (~risk_high) & value_high],
        ["High Risk - High Value", "High Risk - Low Value", "Low Risk - High Value"],
        default="Low Risk - Low Value",
    )
    return predictions


def score_customers(
    features: pd.DataFrame,
    clv_data: pd.DataFrame,
    champion_template: Any,
    champion_cols: List[str],
    champion_name: str,
    champion_threshold: float,
    clv_champion: Any,
    clv_champion_name: str,
    feature_cols: List[str],
    id_col: str,
    target_col: str,
    scenarios: Dict[str, Dict[str, float]],
    paths: Dict[str, Path],
    cut_off_day: int,
) -> pd.DataFrame:
    """Fit final models on all labeled data, score customers, and export all deliverables."""
    X = features[feature_cols].copy()
    y = features[target_col].astype(int).copy()

    final_churn_model = clone(champion_template)
    final_churn_model.fit(features[champion_cols], y)
    p_churn_all = final_churn_model.predict_proba(features[champion_cols])[:, 1]
    predicted_churn = (p_churn_all >= champion_threshold).astype(int)

    clv_pos = clv_data[clv_data["future_revenue_60d"] > 0].copy()
    y_clv_all = np.log1p(clv_pos["future_revenue_60d"])
    final_clv_model = clone(clv_champion)
    final_clv_model.fit(clv_pos[feature_cols], y_clv_all)
    pred_clv_log = final_clv_model.predict(X)
    pred_clv_conditional = np.expm1(np.maximum(pred_clv_log, 0))
    pred_revenue_unconditional = (1 - p_churn_all) * pred_clv_conditional

    predictions = pd.DataFrame({
        id_col: features[id_col].astype(int),
        "actual_churn_flag": y.astype(int),
        "p_churn": p_churn_all,
        "predicted_churn": predicted_churn,
        "predicted_CLV_60d_if_retained": pred_clv_conditional,
        "predicted_revenue_60d_unconditional": pred_revenue_unconditional,
    })
    predictions = calculate_deciles_and_segments(predictions)

    for scenario, params in scenarios.items():
        predictions[f"expected_profit_{scenario}"] = (
            predictions["p_churn"]
            * predictions["predicted_CLV_60d_if_retained"]
            * float(params["retention_lift"])
            * float(params["gross_margin"])
            - float(params["treatment_cost"])
        )
        predictions[f"profitable_to_treat_{scenario}"] = predictions[f"expected_profit_{scenario}"] > 0

    predictions["priority_rank"] = predictions["expected_profit_base"].rank(method="first", ascending=False).astype(int)
    predictions["recommended_treatment_action_base"] = np.where(
        predictions["profitable_to_treat_base"],
        "Candidate for treatment / A-B test",
        np.where(
            predictions["risk_decile"] <= 3,
            "High risk but not profitable under base assumptions",
            "Monitor / no paid treatment",
        ),
    )

    scenario_summary_rows: List[Dict[str, Any]] = []
    for scenario, params in scenarios.items():
        ep_col = f"expected_profit_{scenario}"
        profitable_flag = f"profitable_to_treat_{scenario}"
        positive_ep = predictions.loc[predictions[profitable_flag], ep_col]
        top_risk_ep = predictions.loc[predictions["risk_decile"] <= 3, ep_col]
        scenario_summary_rows.append({
            "scenario": scenario,
            "gross_margin": params["gross_margin"],
            "retention_lift": params["retention_lift"],
            "treatment_cost": params["treatment_cost"],
            "profitable_customer_count": int(predictions[profitable_flag].sum()),
            "profitable_customer_share": float(predictions[profitable_flag].mean()),
            "total_expected_profit_if_target_positive_only": float(positive_ep.sum()) if len(positive_ep) else 0.0,
            "top_30pct_risk_customer_count": int((predictions["risk_decile"] <= 3).sum()),
            "total_expected_profit_if_target_top_30pct_risk": float(top_risk_ep.sum()),
        })

    scenario_profit_summary = pd.DataFrame(scenario_summary_rows)
    scenario_profit_summary.to_csv(paths["models_dir"] / "scenario_profit_summary.csv", index=False)

    voucher_path = paths["voucher_recommendations_csv"]
    if voucher_path.exists():
        vouchers = pd.read_csv(voucher_path)
        if id_col in vouchers.columns:
            vouchers[id_col] = vouchers[id_col].astype(int)
            sort_cols = [id_col]
            if "voucher_recommendation_rank" in vouchers.columns:
                sort_cols.append("voucher_recommendation_rank")
            top_voucher = vouchers.sort_values(sort_cols).groupby(id_col, as_index=False).first()
            keep_cols = [
                id_col,
                "recommended_item",
                "recommended_item_group",
                "predicted_purchase_score",
                "strongest_purchased_anchor_item",
            ]
            keep_cols = [c for c in keep_cols if c in top_voucher.columns]
            predictions = predictions.merge(top_voucher[keep_cols], on=id_col, how="left")

    predictions = predictions.sort_values("priority_rank").reset_index(drop=True)
    predictions.to_csv(paths["models_dir"] / "churn_predictions.csv", index=False)

    high_risk = predictions[predictions["risk_decile"] <= 3].copy()
    high_risk = high_risk.sort_values(["expected_profit_base", "p_churn"], ascending=False)
    high_risk.to_csv(paths["models_dir"] / "high_risk_customers_for_ab_test.csv", index=False)

    predictions.sort_values(["expected_profit_base", "p_churn"], ascending=False).to_csv(
        paths["models_dir"] / "priority_customers_all.csv", index=False
    )

    profitable_candidates = predictions[predictions["profitable_to_treat_base"]].copy()
    profitable_candidates = profitable_candidates.sort_values(["expected_profit_base", "p_churn"], ascending=False)
    profitable_candidates.to_csv(paths["models_dir"] / "profitable_treatment_candidates_base.csv", index=False)

    model_package = {
        "cut_off_day": cut_off_day,
        "feature_cols": feature_cols,
        "champion_churn_model_name": champion_name,
        "champion_threshold": champion_threshold,
        "final_churn_model": final_churn_model,
        "clv_model_name": clv_champion_name,
        "final_clv_model": final_clv_model,
        "scenarios": scenarios,
        "note": "M5 predicts churn risk and value; M6 is responsible for causal A/B test validation.",
    }
    joblib.dump(model_package, paths["models_dir"] / "m5_model_package.pkl")
    joblib.dump(model_package, paths["models_dir"] / "model.pkl")

    return predictions


def save_visual_exports(importance_df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    """Create lightweight report-ready feature importance plot."""
    if importance_df.empty:
        return
    top = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Churn Model Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(paths["visualization_exports_dir"] / "m5_feature_importance_top15.png", dpi=160)
    plt.close()


def write_report_outline(
    metrics_df: pd.DataFrame,
    clv_metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    predictions: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    """Write concise report notes for the final paper."""
    best = metrics_df.iloc[0]
    clv_best = clv_metrics_df.iloc[0]
    feature_top = importance_df.head(5)["feature"].tolist() if not importance_df.empty else []
    base_profitable = int(predictions["profitable_to_treat_base"].sum()) if "profitable_to_treat_base" in predictions else 0

    report_md = f"""# M5 Modeling Report Outline — Churn, CLV, and Expected Profit

## Role in the pipeline
M5 receives the feature table from M4, benchmarks churn models, estimates 60-day customer value, and exports customer-level priority files for M6. M5 does **not** claim causal campaign impact; M6 validates causal lift through A/B testing.

## Methodology
- Input feature table: `models/final_ML_features.csv`.
- `household_key` is retained only as an identifier and excluded from modeling.
- `Primary_Store_ID` is handled as categorical data using one-hot encoding.
- Train/validation/test split is stratified by `churn_flag` to preserve class balance.
- Multiple classifiers are compared: Dummy baseline, Logistic Regression, Random Forest, Extra Trees, XGBoost, and SMOTETomek where available.
- Model selection prioritizes PR-AUC and F2-score rather than accuracy because churn is imbalanced.
- Threshold is tuned on the validation set by maximizing F2-score.
- Customer value is operationalized as predicted 60-day revenue if retained/active, learned from post-cutoff future revenue.
- Expected profit is scenario-based and separates churn risk from treatment eligibility.

## Best churn model
- Selected model: **{best['model']}**
- Validation PR-AUC: **{best['val_PR_AUC']:.4f}**
- Validation F2-score: **{best['val_F2_score']:.4f}**
- Test PR-AUC: **{best['test_PR_AUC']:.4f}**
- Test F2-score: **{best['test_F2_score']:.4f}**
- Tuned threshold: **{best['val_threshold']:.2f}**

## CLV model
- Selected model: **{clv_best['clv_model']}**
- Validation RMSE on log revenue: **{clv_best['val_RMSE_log']:.4f}**
- CLV definition: predicted 60-day future revenue conditional on being retained/active.

## Top churn drivers
{chr(10).join([f'- {f}' for f in feature_top]) if feature_top else '- Native feature importance unavailable for selected model.'}

## Expected profit formula
`Expected Profit_i = p_churn_i × predicted_CLV_60d_if_retained_i × retention_lift × gross_margin - treatment_cost_i`

## Profitability sanity check
Under the base scenario, **{base_profitable}** customers have positive expected treatment profit. High churn risk alone should therefore be treated as an experiment population signal, not as automatic voucher eligibility.

## Files generated
- `models/final_features.parquet`
- `models/model_metrics.csv`
- `models/champion_test_metrics.csv`
- `models/clv_model_metrics.csv`
- `models/feature_importance.csv`
- `models/churn_predictions.csv`
- `models/high_risk_customers_for_ab_test.csv`
- `models/priority_customers_all.csv`
- `models/scenario_profit_summary.csv`
- `models/profitable_treatment_candidates_base.csv`
- `models/model.pkl`

## Suggested concise paper paragraph
We benchmarked multiple churn classifiers, including baseline, linear, bagging, and boosting models. Due to the imbalanced churn label, model selection prioritized Precision-Recall AUC and F2-score, with the final classification threshold tuned on the validation set to emphasize recall. The selected model was then used to score all customers with churn probabilities. To support retention prioritization, we estimated 60-day customer value conditional on retention and combined it with churn probability under scenario-based assumptions for gross margin, treatment cost, and expected retention lift. This produced a ranked customer list for the subsequent A/B testing design, while explicitly flagging that high churn risk alone does not guarantee positive incremental profit.
"""
    (paths["reports_internal_dir"] / "M5_modeling_report_outline.md").write_text(report_md, encoding="utf-8")


def run_m5_pipeline(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Run the full M5 pipeline and return a compact summary."""
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
    test_size = float(cfg["test_size"])
    validation_size = float(cfg["validation_size"])
    f_beta = float(cfg["f_beta"])
    n_estimators = int(cfg["n_estimators"])
    n_jobs = int(cfg["n_jobs"])
    scenarios = config["expected_profit_scenarios"]

    print("[M5] Project root:", project_root)
    features = load_feature_table(paths["feature_table_csv"], id_col, target_col, categorical_cols)
    audit_df, missing_investigation = audit_inputs(features, paths, id_col, target_col, cut_off_day)

    # Keep both the original M4 CSV and the planned final_features.parquet artifact.
    features.to_parquet(paths["models_dir"] / "final_features.parquet", index=False)

    feature_cols = [c for c in features.columns if c not in [id_col, target_col]]
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    split = prepare_splits(features, id_col, target_col, feature_cols, test_size, validation_size, random_state)
    scale_ohe_preprocess, tree_preprocess = build_preprocessors(num_cols, cat_cols)
    scale_pos_weight = float((split["y_train"] == 0).sum() / max((split["y_train"] == 1).sum(), 1))

    model_specs = build_classifier_specs(
        scale_ohe_preprocess,
        tree_preprocess,
        feature_cols,
        num_cols,
        scale_pos_weight,
        random_state,
        n_estimators,
        n_jobs,
    )
    metrics_df, fitted_models = benchmark_classifiers(model_specs, split, f_beta, paths["models_dir"])
    champion_name, champion_threshold, champion_template, champion_cols, importance_df = finalize_champion(
        metrics_df,
        fitted_models,
        split,
        feature_cols,
        paths["models_dir"],
        f_beta,
    )
    save_visual_exports(importance_df, paths)

    clv_data = build_clv_labels(features, paths, id_col, cut_off_day)
    reg_specs = build_regressor_specs(scale_ohe_preprocess, tree_preprocess, random_state, n_estimators, n_jobs)
    clv_metrics_df, clv_models, clv_champion_name, clv_champion = benchmark_clv_models(
        clv_data,
        split,
        feature_cols,
        reg_specs,
        paths["models_dir"],
        id_col,
    )

    predictions = score_customers(
        features,
        clv_data,
        champion_template,
        champion_cols,
        champion_name,
        champion_threshold,
        clv_champion,
        clv_champion_name,
        feature_cols,
        id_col,
        target_col,
        scenarios,
        paths,
        cut_off_day,
    )
    write_report_outline(metrics_df, clv_metrics_df, importance_df, predictions, paths)

    summary = {
        "project_root": str(project_root),
        "feature_rows": int(len(features)),
        "churn_rate": float(features[target_col].mean()),
        "champion_churn_model": champion_name,
        "champion_threshold": float(champion_threshold),
        "validation_PR_AUC": float(metrics_df.iloc[0]["val_PR_AUC"]),
        "validation_F2_score": float(metrics_df.iloc[0]["val_F2_score"]),
        "test_PR_AUC": float(metrics_df.iloc[0]["test_PR_AUC"]),
        "test_F2_score": float(metrics_df.iloc[0]["test_F2_score"]),
        "clv_champion_model": clv_champion_name,
        "base_profitable_customers": int(predictions["profitable_to_treat_base"].sum()),
        "outputs_dir": str(paths["models_dir"]),
    }
    (paths["reports_internal_dir"] / "m5_pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[M5] Pipeline complete.")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M5 churn, CLV, and expected-profit modeling pipeline.")
    parser.add_argument("--config", default="config/paths.yaml", help="Path to YAML config file, relative to project root by default.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_m5_pipeline(args.config)
