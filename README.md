# DDM-Churn-Project
A data-driven churn prediction and retention optimization project using machine learning, market basket analysis, and PSM testing.

## Introduction
In grocery retail, churn is non-contractual and does not have a formal cancellation event. Here, churn is defined as an unusual decline in shopping activity relative to each household's historical purchasing pattern. The workflow spans four stages: data integration and EDA, churn label construction using Inter-Purchase Time (IPT), churn-risk prediction modeling, and coupon-campaign impact assessment using Propensity Score Matching (PSM).

## Dataset
This project uses the Dunnhumby dataset at the household level. The data are relational, spanning eight parquet tables that link transactions, demographics, products, campaigns, coupons, and marketing exposure through shared identifiers such as household_key and PRODUCT_ID. The transaction table is the behavioral backbone, while the remaining tables provide customer, product, and promotional context.

## Repository structure
- notebooks/: Notebook sequence from 01 to 06.
- scripts/: Reusable pipelines for preprocessing, modeling, evaluation, and PSM.
- Data/: Raw, processed, and intermediate datasets.
- models/: Feature tables and model artifacts.
- reports/: final write-ups.
- config/: Project configuration and paths.

## Notebook workflow (run in order)
1. 01_Data_Integration_and_Labeling.ipynb: Integrate raw data and create churn labels.
2. 02_Exploratory_Data_Analysis.ipynb: EDA, distribution checks, and data quality review.
3. 03_Market_Basket.ipynb: Market basket analysis and recommendation artifacts.
4. 04_Feature_Engineering.ipynb: Build modeling features and export feature tables.
5. 05_Predictive_Modeling.ipynb: Train and evaluate churn models, including calibration and explainability.
6. 06_PSM_and_Business_Impact.ipynb: PSM and business impact estimation.

## Scripts overview
- scripts/data_processing.py: CLI pipeline that reproduces notebook 01 preprocessing and churn labeling.
- scripts/feature_engineering.py: Feature engineering workflow used by notebook 04.
- scripts/modeling.py: End-to-end modeling pipeline (training, validation, SHAP, diagnostics).
- scripts/evaluation.py: Evaluation utilities for model performance and ranking metrics.
- scripts/psm_propensity_score.py: Propensity score estimation for PSM.
- scripts/psm_pipeline.py: PSM execution and summary outputs.
- scripts/utils.py: Shared helpers for config and paths.

## Environment setup
1. Create and activate a Python environment (recommended Python 3.10+).
2. Install dependencies:
	pip install -r requirements.txt

If you hit an ImportError while running a notebook or script, install the missing package and add it to requirements.txt. The notebooks already include their import cells, so no manual imports are required beyond running those cells from top to bottom.

## Configuration
All paths and key parameters live in config/paths.yaml. Update this file if you move data folders or want to change modeling settings.

Key inputs referenced in config/paths.yaml include:
- Data/Processed/transactions_master.parquet
- Data/Processed/customer_base_labeled.parquet
- models/final_ML_features.csv
- models/psm_inputs/psm_treatment_flags.csv

## Running scripts (CLI)
Run scripts from the project root. Example commands:

python scripts/data_processing.py --config config/paths.yaml
python scripts/modeling.py --config config/paths.yaml
python scripts/psm_propensity_score.py --config config/paths.yaml

Other scripts are primarily imported by notebooks. Use the notebook flow for the full, step-by-step execution.

## Outputs
Primary outputs are saved under:
- Data/Processed and Data/Intermediate for cleaned data and features
- models/ for artifacts and handoff files
- reports/ for summaries and documentation
