#!/usr/bin/env bash
set -euo pipefail

# Untrack large/customer-level/generated artifacts that should remain local.
# This does not delete local files; it only removes them from Git's index.
# Run from the repository root before pushing to GitHub.

paths=(
  "Data/Raw"
  "Data/Processed"
  "Data/Intermediate"
  "models/model.pkl"
  "models/final_features.parquet"
  "models/final_train_features.parquet"
  "models/final_test_features.parquet"
  "models/final_ML_features.csv"
  "models/churn_predictions.csv"
  "models/priority_customers_all.csv"
  "models/high_risk_customers_for_ab_test.csv"
  "models/discounted_value_labels.csv"
  "models/clv_training_labels.csv"
  "models/profitable_treatment_candidates_base.csv"
  "models/break_even_analysis.csv"
  "models/scenario_sensitivity_grid.csv"
  "models/shap_top_risk_customer_reasons.csv"
  "reports/internal_briefs/m5_pipeline_summary.json"
  "reports/internal_briefs/repo_input_validation.json"
  "reports/internal_briefs/m5_data_audit.csv"
  "reports/internal_briefs/M5_shap_status.json"
  "reports/internal_briefs/M5_v2_change_log.md"
  "reports/internal_briefs/M5_v3_change_log.md"
  "reports/internal_briefs/feature_lineage_audit_template.csv"
  "reports/internal_briefs/multicollinearity_vif.csv"
  "reports/internal_briefs/numeric_feature_correlation_pairs.csv"
  "reports/internal_briefs/value_model_residual_summary.csv"
  "reports/internal_briefs/value_model_decile_diagnostics.csv"
  "reports/internal_briefs/voucher_diversification_audit.csv"
  "reports/internal_briefs/recommendation_merge_audit.csv"
  "reports/internal_briefs/split_stability_runs.csv"
  "reports/internal_briefs/split_stability_summary.csv"
  "reports/internal_briefs/seasonality_audit.csv"
  "reports/internal_briefs/seasonality_window_comparison.csv"
)

for path in "${paths[@]}"; do
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    git rm --cached -r "$path"
  fi
done

echo "Git index cleanup complete. Review with: git status --short"
