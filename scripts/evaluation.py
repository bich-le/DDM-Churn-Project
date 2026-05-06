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


def best_fbeta_threshold(
    y_true: Iterable[int], proba: Iterable[float], beta: float = 2.0
) -> Tuple[float, float]:
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


def evaluate_proba(
    y_true: Iterable[int], proba: Iterable[float], threshold: float, beta: float = 2.0
) -> Dict[str, Any]:
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


def evaluate_classifier(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    threshold: float,
    beta: float = 2.0,
) -> Dict[str, Any]:
    """Evaluate a fitted probabilistic binary classifier at a chosen threshold."""
    proba = model.predict_proba(X_eval)[:, 1]
    return evaluate_proba(y_eval, proba, threshold=threshold, beta=beta)


def calibration_by_decile(
    y_true: Iterable[int], proba: Iterable[float], n_bins: int = 10
) -> pd.DataFrame:
    """Return a decile-level reliability table for predicted probabilities."""
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "proba": np.asarray(proba).astype(float),
        }
    )
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
    out["calibration_gap"] = (
        out["mean_predicted_probability"] - out["actual_churn_rate"]
    )
    return out


def ranking_decile_performance(
    y_true: Iterable[int],
    score: Iterable[float],
    n_deciles: int = 10,
    score_name: str = "score",
) -> pd.DataFrame:
    """Evaluate how well a score ranks churn cases by decile.

    Decile 1 is always the highest-score / highest-priority group. The table is
    designed for business review: it shows whether the top-scored customers
    actually churn at a meaningfully higher rate than the baseline churn rate.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            score_name: np.asarray(score).astype(float),
        }
    )
    if df.empty:
        return pd.DataFrame()
    baseline = float(df["y_true"].mean())
    total_churn = max(int(df["y_true"].sum()), 1)
    # Rank first to avoid qcut failures when many customers have tied scores.
    df["ranking_decile"] = pd.qcut(
        df[score_name].rank(method="first", ascending=False),
        q=n_deciles,
        labels=list(range(1, n_deciles + 1)),
    ).astype(int)
    out = (
        df.groupby("ranking_decile")
        .agg(
            customer_count=("y_true", "size"),
            churn_count=("y_true", "sum"),
            churn_rate=("y_true", "mean"),
            min_score=(score_name, "min"),
            mean_score=(score_name, "mean"),
            max_score=(score_name, "max"),
        )
        .reset_index()
        .sort_values("ranking_decile")
    )
    out["baseline_churn_rate"] = baseline
    out["lift_vs_baseline"] = out["churn_rate"] / baseline if baseline > 0 else np.nan
    out["cumulative_customers"] = out["customer_count"].cumsum()
    out["cumulative_churn"] = out["churn_count"].cumsum()
    out["cumulative_precision"] = out["cumulative_churn"] / out["cumulative_customers"]
    out["cumulative_recall"] = out["cumulative_churn"] / total_churn
    return out


def top_k_precision_summary(
    y_true: Iterable[int],
    score: Iterable[float],
    top_k_shares: Iterable[float] = (0.05, 0.10, 0.20),
    score_name: str = "score",
) -> pd.DataFrame:
    """Return Precision@K, Recall@K, and Lift@K for top-scored customers."""
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            score_name: np.asarray(score).astype(float),
        }
    )
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(score_name, ascending=False).reset_index(drop=True)
    baseline = float(df["y_true"].mean())
    total_churn = max(int(df["y_true"].sum()), 1)
    rows: List[Dict[str, Any]] = []
    n_total = len(df)
    for share in top_k_shares:
        share = float(share)
        n = int(np.ceil(n_total * share))
        n = min(max(n, 1), n_total)
        top = df.head(n)
        precision = float(top["y_true"].mean())
        churn_count = int(top["y_true"].sum())
        rows.append(
            {
                "score_name": score_name,
                "top_k_share": share,
                "top_k_customer_count": n,
                "baseline_churn_rate": baseline,
                "precision_at_k": precision,
                "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
                "churn_count_at_k": churn_count,
                "recall_at_k": churn_count / total_churn,
                "min_score_in_top_k": float(top[score_name].min()),
                "mean_score_in_top_k": float(top[score_name].mean()),
                "max_score_in_top_k": float(top[score_name].max()),
            }
        )
    return pd.DataFrame(rows)


def profit_threshold_analysis(
    predictions: pd.DataFrame,
    expected_profit_cols: Iterable[str],
    y_col: str = "actual_churn_flag",
    p_col: str = "p_churn_calibrated",
    top_k_shares: Iterable[float] = (0.05, 0.10, 0.20),
    min_profit: float = 0.0,
) -> pd.DataFrame:
    """Summarize business treatment rules based on expected incremental profit.

    This does not claim causal lift. It translates scenario assumptions into
    candidate selection rules so the team can compare paid-treatment targeting
    against simple churn-risk ranking.
    """
    rows: List[Dict[str, Any]] = []
    n_total = len(predictions)
    baseline = float(predictions[y_col].mean()) if y_col in predictions else np.nan
    for col in expected_profit_cols:
        if col not in predictions.columns:
            continue
        scenario = col.replace("expected_incremental_profit_", "")
        ordered = predictions.sort_values(col, ascending=False).reset_index(drop=True)
        positive = predictions[predictions[col] > min_profit]
        selections = [("profit_positive", positive)]
        for share in top_k_shares:
            n = min(max(int(np.ceil(n_total * float(share))), 1), n_total)
            selections.append(
                (
                    f"top_{int(float(share) * 100)}pct_by_expected_profit",
                    ordered.head(n),
                )
            )
        for rule, selected in selections:
            selected_count = len(selected)
            selected_churn_rate = (
                float(selected[y_col].mean())
                if selected_count and y_col in selected
                else np.nan
            )
            rows.append(
                {
                    "scenario": scenario,
                    "expected_profit_column": col,
                    "selection_rule": rule,
                    "selected_customer_count": int(selected_count),
                    "selected_customer_share": float(selected_count / n_total)
                    if n_total
                    else np.nan,
                    "baseline_churn_rate": baseline,
                    "selected_churn_rate": selected_churn_rate,
                    "lift_vs_baseline": selected_churn_rate / baseline
                    if baseline and not np.isnan(selected_churn_rate)
                    else np.nan,
                    "total_expected_incremental_profit": float(selected[col].sum())
                    if selected_count
                    else 0.0,
                    "mean_expected_incremental_profit": float(selected[col].mean())
                    if selected_count
                    else np.nan,
                    "min_expected_incremental_profit": float(selected[col].min())
                    if selected_count
                    else np.nan,
                    "max_expected_incremental_profit": float(selected[col].max())
                    if selected_count
                    else np.nan,
                    "min_p_churn_calibrated": float(selected[p_col].min())
                    if selected_count and p_col in selected
                    else np.nan,
                    "mean_p_churn_calibrated": float(selected[p_col].mean())
                    if selected_count and p_col in selected
                    else np.nan,
                    "max_p_churn_calibrated": float(selected[p_col].max())
                    if selected_count and p_col in selected
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def regression_metrics(
    y_true_log: Iterable[float], pred_log: Iterable[float], prefix: str
) -> Dict[str, float]:
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
        f"{prefix}_MAE_revenue": float(
            mean_absolute_error(np.expm1(y_true_log), np.expm1(pred_log))
        ),
    }


def get_feature_names_from_pipeline(
    pipeline: Any, original_columns: List[str]
) -> List[str]:
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
