"""Evaluation helpers for M5 churn, 60-day value, calibration, and profit modeling.

These functions deliberately contain no project-specific paths so they can be
used from both the CLI pipeline and notebooks.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder


def make_one_hot_encoder() -> OneHotEncoder:
    """Return a OneHotEncoder compatible with multiple scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def best_fbeta_threshold(y_true: Iterable[int], proba: Iterable[float], beta: float = 2.0) -> Tuple[float, float]:
    """Search for the threshold that maximizes F-beta on validation data."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        rows.append((float(threshold), float(score)))
    return max(rows, key=lambda x: x[1])


def evaluate_proba(y_true: Iterable[int], proba: Iterable[float], threshold: float, beta: float = 2.0) -> Dict[str, Any]:
    """Evaluate binary-class probabilities at a chosen classification threshold."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    pred = (proba >= threshold).astype(int)
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, proba))
    else:
        roc_auc = np.nan
    metrics: Dict[str, Any] = {
        "PR_AUC": float(average_precision_score(y_true, proba)),
        "ROC_AUC": roc_auc,
        "F2_score": float(fbeta_score(y_true, pred, beta=beta, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "predicted_positive_rate": float(pred.mean()),
        "mean_predicted_probability": float(proba.mean()),
        "actual_positive_rate": float(y_true.mean()),
        "calibration_gap_mean_minus_actual": float(proba.mean() - y_true.mean()),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return metrics


def evaluate_classifier(model: Any, X_eval: pd.DataFrame, y_eval: pd.Series, threshold: float, beta: float = 2.0) -> Dict[str, Any]:
    """Evaluate a fitted probabilistic binary classifier at a chosen threshold."""
    proba = model.predict_proba(X_eval)[:, 1]
    return evaluate_proba(y_eval, proba, threshold=threshold, beta=beta)


def calibration_by_decile(y_true: Iterable[int], proba: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Return a decile-level reliability table for predicted probabilities."""
    df = pd.DataFrame({"y_true": np.asarray(y_true).astype(int), "proba": np.asarray(proba).astype(float)})
    # qcut can fail when many probabilities tie; rank first for stable equal-sized bins.
    df["probability_decile"] = pd.qcut(
        df["proba"].rank(method="first"),
        q=n_bins,
        labels=list(range(n_bins, 0, -1)),
    ).astype(int)
    out = (
        df.groupby("probability_decile")
        .agg(
            customer_count=("y_true", "size"),
            mean_predicted_probability=("proba", "mean"),
            actual_churn_rate=("y_true", "mean"),
            churn_count=("y_true", "sum"),
        )
        .reset_index()
        .sort_values("probability_decile")
    )
    out["calibration_gap"] = out["mean_predicted_probability"] - out["actual_churn_rate"]
    return out


def regression_metrics(y_true_log: Iterable[float], pred_log: Iterable[float], prefix: str) -> Dict[str, float]:
    """Evaluate log-revenue regression and revenue-scale MAE."""
    y_true_log = np.asarray(y_true_log)
    pred_log = np.maximum(np.asarray(pred_log), 0)
    if len(y_true_log) == 0:
        return {
            f"{prefix}_RMSE_log": np.nan,
            f"{prefix}_MAE_log": np.nan,
            f"{prefix}_R2_log": np.nan,
            f"{prefix}_MAE_revenue": np.nan,
        }
    return {
        f"{prefix}_RMSE_log": float(np.sqrt(mean_squared_error(y_true_log, pred_log))),
        f"{prefix}_MAE_log": float(mean_absolute_error(y_true_log, pred_log)),
        f"{prefix}_R2_log": float(r2_score(y_true_log, pred_log)),
        f"{prefix}_MAE_revenue": float(mean_absolute_error(np.expm1(y_true_log), np.expm1(pred_log))),
    }


def get_feature_names_from_pipeline(pipeline: Any, original_columns: List[str]) -> List[str]:
    """Best-effort extraction of transformed feature names from a sklearn pipeline."""
    if not hasattr(pipeline, "named_steps") or "preprocess" not in pipeline.named_steps:
        return original_columns
    preprocess = pipeline.named_steps["preprocess"]
    try:
        names = preprocess.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        return original_columns


def extract_feature_importance(model: Any, original_columns: List[str]) -> pd.DataFrame:
    """Extract native feature importance or coefficients from the final estimator."""
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
        feature_names = get_feature_names_from_pipeline(model, original_columns)
    else:
        estimator = model
        feature_names = original_columns

    if estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(np.abs(estimator.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(feature_names), len(values))
    out = pd.DataFrame({"feature": feature_names[:n], "importance": values[:n]})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
