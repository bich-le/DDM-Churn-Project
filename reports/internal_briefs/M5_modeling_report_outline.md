# M5 Modeling Report Outline — v3

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
- XGBoost benchmarking is enabled in the full-quality configuration; it should be compared under the same calibration and validation protocol as the other candidates.
- SHAP outputs are used for model explainability, but they are predictive explanations, not causal claims.

## Churn model
- Selected reporting model: **Logistic Regression balanced**
- Calibration selected: **isotonic**
- Threshold on calibrated probability: **0.07**
- Test PR-AUC: **0.3185**
- Test F2-score: **0.4721**
- Test Brier score: **0.0933**
- Test mean predicted probability: **0.1213** vs actual churn rate **0.1200**

The selected model should be interpreted as the reporting/champion model for calibrated decision support, not as proof that it is statistically superior to every alternative. With a small sample size and unstable positive-class counts, differences across candidate models may fall within split-level variance.

## Two-part discounted value model
- Future-active model: **Active Random Forest (isotonic)**
- Active-model test Brier score: **0.0715**
- Conditional value model: **Random Forest Regressor**
- Conditional value target: **log1p(discounted_future_revenue_60d | future_active=1)**
- Test RMSE_log: **0.8851**

## Expected incremental profit and candidate ranking
`Expected Incremental Profit_i = p_churn_calibrated_i × save_rate_given_treatment × predicted_discounted_value_60d_if_active_i × gross_margin - treatment_cost_i`

The profit formula intentionally uses `predicted_discounted_value_60d_if_active`, not the unconditional `p_future_active × value_if_active`, to avoid double-counting the active/churn probability. Under the base scenario, **0** customers have positive expected incremental treatment profit.

If no customers have positive expected profit under the base scenario, `priority_rank` falls back to churn-risk ranking for A/B test candidate selection. In that case, profit ranking should not be interpreted as treatment eligibility; it is only a scenario diagnostic.

## SHAP top drivers
- Recency_Capped
- Inactive_Days_Ratio
- Camp_Count_TypeA
- Campaigns_Last_30D
- Mailer_Responsiveness

## Important limitations
- M5 uses M4's current feature table as-is. A separate M4 feature lineage audit is still recommended.
- Discounted 60-day value is not true lifetime CLV.
- SHAP and feature importance are associations, not causal treatment effects.
- Seasonality is audited, but not fully modeled as a time-series forecasting problem.
