# PSM Input Files

This folder stores **input files required by M5 to support M6's Propensity Score Matching (PSM)**.

## Required file from M4

M4 should provide:

```text
models/psm_inputs/psm_treatment_flags.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `household_key` | Household identifier. Must match `models/final_ML_features.csv`. |
| `is_treated` | `1` if the household used/received coupon or mailer before day 651, else `0`. |
| `treatment_source` | Short description, e.g. `coupon_or_mailer_before_cutoff`. |
| `treatment_cutoff_day` | Should be `651` for the current project setup. |

## Important note

`psm_treatment_flags.csv` is an **input** for M5's propensity-score script, not an M6 handoff output.

M5 will read this file, merge it with `models/final_ML_features.csv`, run Logistic Regression, and export:

```text
models/m6_handoff/propensity_scores_for_psm.csv
```

## Placeholder / dry-run rule

A dummy treatment flag file can be generated for code testing only:

```bash
python scripts/create_dummy_psm_treatment_flags.py --config config/paths.yaml --overwrite
```

Do **not** use dummy propensity scores in the final report. Replace the dummy file with M4's real treatment flags before final PSM analysis.
