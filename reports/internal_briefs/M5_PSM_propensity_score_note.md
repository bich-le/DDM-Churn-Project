# M5 PSM Propensity Score Handoff

M5 estimates `propensity_score = P(is_treated = 1 | observed pre-outcome covariates)` for M6's Propensity Score Matching workflow.

- Input treatment flag file: `models\psm_inputs\psm_treatment_flags.csv`
- Output for M6: `models\m6_handoff\propensity_scores_for_psm.csv`
- Covariates used: `models\diagnostics\psm_covariates_used.csv`
- Model summary: `models\diagnostics\psm_propensity_model_summary.csv`

## M4 dependency

Replace the dummy/placeholder treatment file with M4's real treatment flag file before final PSM reporting. Required columns: `household_key`, `is_treated`, `treatment_source`, `treatment_cutoff_day`.

## Interpretation warning

This is not a randomized A/B test. PSM can balance observed covariates, but unobserved confounding may remain. Write the conclusion as association after matching, not definitive causal proof.
