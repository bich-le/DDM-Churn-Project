# M5 Imbalance and Resampling Audit

## Rule
Validation and test sets are never resampled. Test data must preserve the real churn distribution.

## Current pipeline
- Train/validation/test split is created before any optional resampling experiment.
- Main churn models use class weighting and probability calibration, not SMOTE, because business formulas require meaningful probabilities.
- Optional SMOTETomek baseline enabled: `False`.
- If enabled, SMOTETomek is wrapped inside an `imblearn.pipeline.Pipeline`, so resampling happens only inside the training pipeline.

## Why this matters
SMOTE before train/test split would leak synthetic information into validation/test and inflate model quality. M5 therefore treats SMOTE as an optional benchmark, never as a pre-processing artifact from M4.
