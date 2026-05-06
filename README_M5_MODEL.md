# README — M5 Model Pipeline v3 + diagnostics

This README explains the M5 pipeline for a beginner. M5 builds churn-risk, discounted value, expected-profit rankings, and diagnostic checks for the A/B testing step.

## 1. What M5 outputs

For each household, M5 outputs:

| Output | Meaning |
|---|---|
| `p_churn_calibrated` | Calibrated churn-risk probability under the project churn definition |
| `p_future_active` | Probability that the household will generate revenue in the 60-day prediction window |
| `predicted_discounted_value_60d_if_active` | Discounted 60-day revenue if the household is active |
| `predicted_expected_discounted_value_60d` | Two-part expected value = `p_future_active × value_if_active` |
| `expected_incremental_profit_*` | Scenario-based incremental profit from targeting |
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

Success metrics:

| Metric | Meaning |
|---|---|
| PR-AUC | Ranking quality for rare churn cases |
| F2-score | Churn classification metric that prioritizes recall over precision |
| Recall | Share of actual churners caught |
| Precision | Share of predicted churn-risk customers who actually churn |
| Brier score | Probability quality/calibration |
| Precision@Top K / Lift@Top K | Whether the highest-ranked customers are meaningfully better than random targeting |

### Probability calibration
The champion churn model is calibrated using validation data. The pipeline compares raw, sigmoid, and isotonic probabilities, then selects the best validation Brier score. Business formulas use `p_churn_calibrated`, not raw weighted-model probability.

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

This is scenario-based. The save rate, margin, and treatment cost are assumptions until M6 validates lift with A/B testing. The pipeline also exports profit-threshold and top-K diagnostics, because a retention campaign usually targets a prioritized subset rather than every predicted churner.

## 3. Data leakage and SMOTE rules

- M5 assumes M4's delivered features are computed before `cut_off_day`.
- `models/feature_lineage_audit_template.csv` is generated so M4 can confirm source table and time window for each feature.
- M5 does not modify M4 features.
- Validation and test sets are never resampled.
- Optional SMOTETomek enabled in this run: `False`.
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

## 6. Additional model diagnostics

| File | Why it matters |
|---|---|
| `models/multicollinearity_vif.csv` | Checks whether numeric features duplicate each other, important for Logistic/Ridge interpretation |
| `models/numeric_feature_correlation_pairs.csv` | Lists numeric feature pairs with high correlation |
| `models/value_model_residual_summary.csv` | Shows whether discounted-value predictions are biased overall |
| `models/value_model_decile_diagnostics.csv` | Shows value-model error by predicted value decile |
| `models/feature_lineage_audit_template.csv` | M4/M5 checklist for feature-window leakage review |
| `models/voucher_diversification_audit.csv` | Checks whether RecSys offers are available/diversified for campaign design |
| `models/split_stability_runs.csv` | Repeated split robustness check; do not cherry-pick the best seed |
| `models/split_stability_summary.csv` | Mean/std/min/max of key metrics across repeated splits |

## 7. How files interact

```text
config/paths.yaml
  -> scripts/utils.py resolves paths
  -> scripts/modeling.py runs the full pipeline
  -> scripts/evaluation.py provides metrics/helpers
  -> models/final_ML_features.csv is the M4 handoff
  -> Data/Processed/transactions_master.parquet builds discounted value labels and seasonality audit
  -> models/*.csv contains outputs for M5/M6
```

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
- `models/shap_global_importance.csv`
- `models/shap_top_risk_customer_reasons.csv`
- `models/seasonality_audit.csv`
- `models/seasonality_window_comparison.csv`
- `models/model.pkl`
- `models/multicollinearity_vif.csv`
- `models/numeric_feature_correlation_pairs.csv`
- `models/value_model_residual_summary.csv`
- `models/value_model_decile_diagnostics.csv`
- `models/feature_lineage_audit_template.csv`
- `models/voucher_diversification_audit.csv`
- `models/split_stability_runs.csv`
- `models/split_stability_summary.csv`

## 10. What to say in the report

Use wording like:

> We estimate calibrated churn risk and two-part discounted 60-day customer value, then combine them with scenario-based save rate, gross margin, and treatment cost to prioritize customers for A/B testing.

Avoid saying:

- “This is true lifetime CLV.”
- “SHAP proves why the customer churned.”
- “High-risk customers should automatically receive vouchers.”
