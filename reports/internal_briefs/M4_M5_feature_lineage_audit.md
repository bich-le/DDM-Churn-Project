# M4 → M5 Feature Lineage Audit Scaffold

## Purpose

This file documents the current M5 assumption about the M4 feature table. M5 uses the delivered feature table as-is and does **not** rebuild M4 features. Therefore, M5 cannot fully prove upstream feature lineage from the final table alone.

## Current cut-off assumption

- Observation window expected for model features: `DAY < 651`
- Prediction/value window used for labels and value targets: `DAY >= 651`

## Files generated

- `models/feature_lineage_audit_template.csv`

## How to use this audit

M4 should confirm each feature's actual source table and time window. Features marked as temporal, promotion/campaign, or purchase-cadence related receive extra caution because they are more likely to accidentally use post-cutoff information.

## Current status

- Total columns audited: 27
- Columns requiring M4 confirmation: 25
- Medium-risk feature-lineage rows: 25

## Important limitation

This scaffold is not proof that all features are leakage-free. It is a handoff checklist for M4/M5 to close before final submission.
