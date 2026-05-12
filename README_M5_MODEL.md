# README — M5 Model Pipeline v3

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

Current config uses `cv_folds: 5` and `tuning_n_iter: 10` for randomized hyperparameter search. Tree baselines and XGBoost are enabled in the full-quality configuration. The search is still not exhaustive, but the XGBoost search space now includes substantially larger `n_estimators` values instead of the earlier lightweight 10/20-tree grid.

Success metrics:

| Metric | Meaning |
|---|---|
| PR-AUC | Ranking quality for rare churn cases |
| F2-score | Churn classification metric that prioritizes recall over precision |
| Recall | Share of actual churners caught |
| Precision | Share of predicted churn-risk customers who actually churn |
| Brier score | Probability quality/calibration |

### Probability calibration
The champion churn model is calibrated using validation data. The pipeline compares raw, sigmoid, and isotonic probabilities. Calibration selection is multi-objective: it first keeps methods whose validation Brier score is close to the best-Brier method, then chooses the highest validation PR-AUC among those methods. This avoids selecting an isotonic model that has slightly better Brier score but destroys ranking resolution through large flat probability plateaus. Business formulas use `p_churn_calibrated`, not raw weighted-model probability.

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
- Optional SMOTETomek enabled in this run: `False`.
- If SMOTETomek is enabled, it is inside an imbalanced-learn Pipeline after the split, not applied to the full dataset.

## 4. SHAP explainability

M5 exports SHAP outputs when the `shap` package is available:

| File | Meaning |
|---|---|
| `models/reports/shap_global_importance.csv` | Global mean absolute SHAP drivers |
| `models/diagnostics/shap_top_risk_customer_reasons.csv` | Local explanations for top-risk customers |
| `visualization/exports/m5_shap_global_importance.png` | Report-ready SHAP importance chart |

Important: SHAP explains model predictions. It does not prove causal treatment effects.

## 5. Seasonality audit

M5 adds a lightweight seasonality check:

| File | Meaning |
|---|---|
| `models/diagnostics/seasonality_audit.csv` | Weekly sales, transactions, and active households |
| `models/diagnostics/seasonality_window_comparison.csv` | Prediction-window revenue compared with recent pre-cutoff windows |
| `visualization/exports/m5_weekly_revenue_trend.png` | Weekly sales trend with cut-off marker |

This does not fully model seasonality, but it documents whether the 60-day prediction window looks unusual.

## 6. How files interact

```text
config/paths.yaml
  -> scripts/utils.py resolves paths
  -> scripts/modeling.py runs the full pipeline
  -> scripts/evaluation.py provides metrics/helpers
  -> models/final_ML_features_linear.csv and models/final_ML_features_tree.csv are the M4 handoff tables
  -> Data/Processed/transactions_master.parquet builds discounted value labels and seasonality audit
  -> models/reports/*.csv contains report summaries
  -> models/m6_handoff/*.csv contains customer-level files for M6
  -> models/diagnostics/*.csv contains internal M5 audit outputs
  -> models/artifacts/* contains local model artifacts
```

## 7. GitHub/data governance note

Raw/processed/intermediate Dunnhumby data and customer-level generated outputs are not intended to be pushed to a public GitHub repository. See `Data/README.md` for local data placement. The repository should track code, configuration, lightweight metric summaries, and documentation; customer-level predictions and heavy artifacts should be regenerated locally.

## 8. How to run

```bash
pip install -r requirements.txt
python -m py_compile scripts/*.py
python scripts/modeling.py --config config/paths.yaml
```

## 9. Main output folders

M5 outputs are separated by purpose. This keeps the repo readable and prevents M6 handoff files, report summaries, diagnostics, and model artifacts from being mixed together.

### `models/reports/` — report/review summaries

- `models/reports/model_metrics.csv`
- `models/reports/champion_test_metrics.csv`
- `models/reports/calibration_summary.csv`
- `models/reports/top_k_precision_summary.csv`
- `models/reports/ranking_decile_performance.csv`
- `models/reports/scenario_profit_summary.csv`
- `models/reports/profit_threshold_analysis.csv`
- `models/reports/value_model_metrics.csv`
- `models/reports/shap_global_importance.csv`

### `models/m6_handoff/` — files for M6 A/B testing

- `models/m6_handoff/high_risk_customers_for_ab_test.csv`
- `models/m6_handoff/priority_customers_all.csv`
- `models/m6_handoff/churn_predictions.csv`
- `models/m6_handoff/recommendation_merge_audit.csv`

### `models/diagnostics/` — internal M5 audits/debug files

- `models/diagnostics/model_tuning_results.csv`
- `models/diagnostics/model_cv_summary.csv`
- `models/diagnostics/calibration_by_decile.csv`
- `models/diagnostics/calibration_flatspot_diagnostics.csv`
- `models/diagnostics/active_model_metrics.csv`
- `models/diagnostics/active_churn_target_overlap_audit.csv`
- `models/diagnostics/discounted_value_labels.csv`
- `models/diagnostics/scenario_sensitivity_grid.csv`
- `models/diagnostics/break_even_analysis.csv`
- `models/diagnostics/shap_top_risk_customer_reasons.csv`
- `models/diagnostics/seasonality_audit.csv`
- `models/diagnostics/seasonality_window_comparison.csv`

### `models/artifacts/` — local model artifacts

- `models/artifacts/model.pkl`
- `models/artifacts/final_features.parquet`

M5 does not automatically move upstream files owned by M3/M4.

## 10. What to say in the report

Use wording like:

> We estimate calibrated churn risk and two-part discounted 60-day customer value, then combine them with scenario-based save rate, gross margin, and treatment cost to prioritize customers for A/B testing.

Avoid saying:

- “This is true lifetime CLV.”
- “SHAP proves why the customer churned.”
- “High-risk customers should automatically receive vouchers.”
