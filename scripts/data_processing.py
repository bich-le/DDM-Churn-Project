"""Project preprocessing pipeline for integrated churn-ready datasets.

This script consolidates the preprocessing logic from
`notebooks/01_Data_Integration_and_Labeling.ipynb` into a reusable CLI:

1. Load raw parquet tables from `Data/Raw`
2. Clean transaction rows and harmonize key dtypes
3. Create churn labels with a personalized IPT threshold
4. Assemble processed tables and export them to `Data/Processed`

Run:
    python scripts/data_processing.py --config config/paths.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.utils import ensure_project_structure, find_project_root, load_config, resolve_project_paths


RAW_DATASETS = {
    "transaction_data": "transaction_data.parquet",
    "hh_demographic": "hh_demographic.parquet",
    "product": "product.parquet",
    "campaign_table": "campaign_table.parquet",
    "campaign_desc": "campaign_desc.parquet",
    "coupon": "coupon.parquet",
    "coupon_redempt": "coupon_redempt.parquet",
    "causal_data": "causal_data.parquet",
}

DEMO_COLS = [
    "AGE_DESC",
    "MARITAL_STATUS_CODE",
    "INCOME_DESC",
    "HOMEOWNER_DESC",
    "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC",
    "KID_CATEGORY_DESC",
]

STRING_CAST_MAP = {
    "transactions": ["household_key", "BASKET_ID", "PRODUCT_ID", "STORE_ID"],
    "hh_demographic": ["household_key"],
    "product": ["PRODUCT_ID"],
    "campaign_table": ["household_key"],
    "coupon": ["COUPON_UPC", "PRODUCT_ID"],
    "coupon_redempt": ["household_key", "COUPON_UPC"],
    "causal_data": ["PRODUCT_ID", "STORE_ID"],
}

EXCLUDE_DEPARTMENTS = ["KIOSK-GAS", "MISC SALES TRAN"]
MAX_VALID_QUANTITY = 150
DATA_END_DAY = 711


def load_raw_datasets(data_raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the raw parquet inputs required by the preprocessing pipeline."""
    datasets: Dict[str, pd.DataFrame] = {}
    for name, filename in RAW_DATASETS.items():
        path = data_raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw dataset: {path}")
        datasets[name] = pd.read_parquet(path)
    return datasets


def cast_key_columns(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Standardize join-key dtypes to string across all relevant tables."""
    casted = {name: df.copy() for name, df in datasets.items()}

    casted["transaction_data"][STRING_CAST_MAP["transactions"]] = casted["transaction_data"][
        STRING_CAST_MAP["transactions"]
    ].astype(str)
    casted["hh_demographic"][STRING_CAST_MAP["hh_demographic"]] = casted["hh_demographic"][
        STRING_CAST_MAP["hh_demographic"]
    ].astype(str)
    casted["product"][STRING_CAST_MAP["product"]] = casted["product"][STRING_CAST_MAP["product"]].astype(str)
    casted["campaign_table"][STRING_CAST_MAP["campaign_table"]] = casted["campaign_table"][
        STRING_CAST_MAP["campaign_table"]
    ].astype(str)
    casted["coupon"][STRING_CAST_MAP["coupon"]] = casted["coupon"][STRING_CAST_MAP["coupon"]].astype(str)
    casted["coupon_redempt"][STRING_CAST_MAP["coupon_redempt"]] = casted["coupon_redempt"][
        STRING_CAST_MAP["coupon_redempt"]
    ].astype(str)
    casted["causal_data"][STRING_CAST_MAP["causal_data"]] = casted["causal_data"][
        STRING_CAST_MAP["causal_data"]
    ].astype(str)

    return casted


def clean_transactions(transaction_data: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    """Filter invalid rows and remove known non-grocery and bulk anomalies."""
    transactions_cleaned = transaction_data[
        (transaction_data["SALES_VALUE"] > 0) & (transaction_data["QUANTITY"] > 0)
    ].copy()

    exclude_product_ids = product.loc[product["DEPARTMENT"].isin(EXCLUDE_DEPARTMENTS), "PRODUCT_ID"]

    transactions_cleaned = transactions_cleaned[
        (~transactions_cleaned["PRODUCT_ID"].isin(exclude_product_ids))
        & (transactions_cleaned["QUANTITY"] <= MAX_VALID_QUANTITY)
    ].copy()

    return transactions_cleaned


def build_customer_churn(transactions_cleaned: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Create churn labels using a personalized IPT threshold with global fallback."""
    unique_purchases = (
        transactions_cleaned[["household_key", "DAY"]]
        .drop_duplicates()
        .sort_values(["household_key", "DAY"])
        .copy()
    )
    unique_purchases["prev_DAY"] = unique_purchases.groupby("household_key")["DAY"].shift(1)
    unique_purchases["IPT"] = unique_purchases["DAY"] - unique_purchases["prev_DAY"]

    customer_churn = (
        unique_purchases.groupby("household_key")
        .agg(
            mean_IPT=("IPT", "mean"),
            std_IPT=("IPT", "std"),
            last_purchase_day=("DAY", "max"),
        )
        .reset_index()
    )

    global_threshold = float(unique_purchases["IPT"].mean() + 2 * unique_purchases["IPT"].std())
    customer_churn["personalized_threshold"] = customer_churn["mean_IPT"] + 2 * customer_churn["std_IPT"]
    customer_churn["personalized_threshold"] = customer_churn["personalized_threshold"].fillna(global_threshold)
    customer_churn["recency"] = DATA_END_DAY - customer_churn["last_purchase_day"]
    customer_churn["is_churn"] = (
        customer_churn["recency"] > customer_churn["personalized_threshold"]
    ).astype(int)

    return customer_churn, global_threshold


def assemble_processed_tables(
    datasets: Dict[str, pd.DataFrame],
    transactions_cleaned: pd.DataFrame,
    customer_churn: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Synchronize all linked tables and create final processed outputs."""
    product = datasets["product"]
    hh_demographic = datasets["hh_demographic"]
    campaign_table = datasets["campaign_table"]
    campaign_desc = datasets["campaign_desc"]
    coupon = datasets["coupon"]
    coupon_redempt = datasets["coupon_redempt"]

    valid_households = set(transactions_cleaned["household_key"].unique())
    valid_products = set(transactions_cleaned["PRODUCT_ID"].unique())

    product_clean = product[product["PRODUCT_ID"].isin(valid_products)].copy()
    hh_demographic_clean = hh_demographic[hh_demographic["household_key"].isin(valid_households)].copy()
    campaign_table_clean = campaign_table[campaign_table["household_key"].isin(valid_households)].copy()
    coupon_redempt_clean = coupon_redempt[coupon_redempt["household_key"].isin(valid_households)].copy()
    coupon_clean = coupon[coupon["PRODUCT_ID"].isin(valid_products)].copy()

    product_subset = product_clean[["PRODUCT_ID", "DEPARTMENT", "BRAND", "COMMODITY_DESC"]].copy()
    transactions_master = transactions_cleaned.merge(product_subset, on="PRODUCT_ID", how="left")

    rfm_metrics = (
        transactions_cleaned.groupby("household_key")
        .agg(Frequency=("BASKET_ID", "nunique"), Monetary=("SALES_VALUE", "sum"))
        .reset_index()
    )

    customer_base_labeled = customer_churn.merge(rfm_metrics, on="household_key", how="left")
    customer_base_labeled = customer_base_labeled.merge(
        hh_demographic_clean,
        on="household_key",
        how="left",
    )
    customer_base_labeled[DEMO_COLS] = customer_base_labeled[DEMO_COLS].fillna("Unknown")

    return {
        "customer_base_labeled": customer_base_labeled,
        "transactions_master": transactions_master,
        "campaign_table_clean": campaign_table_clean,
        "campaign_desc_clean": campaign_desc.copy(),
        "coupon_clean": coupon_clean,
        "demographics_imputed": hh_demographic_clean,
        "coupon_redempt_clean": coupon_redempt_clean,
    }


def export_processed_tables(processed_tables: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    """Write processed tables to parquet files and return exported file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_files: Dict[str, str] = {}

    for name, df in processed_tables.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        exported_files[name] = str(path)

    return exported_files


def run_preprocessing(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Run the end-to-end preprocessing pipeline and return summary metadata."""
    project_root = find_project_root()
    config = load_config(config_path=config_path, project_root=project_root)
    paths = resolve_project_paths(config, project_root)
    ensure_project_structure(paths)

    raw_datasets = load_raw_datasets(paths["data_raw_dir"])
    casted_datasets = cast_key_columns(raw_datasets)
    transactions_cleaned = clean_transactions(
        casted_datasets["transaction_data"],
        casted_datasets["product"],
    )
    customer_churn, global_threshold = build_customer_churn(transactions_cleaned)
    processed_tables = assemble_processed_tables(casted_datasets, transactions_cleaned, customer_churn)
    exported_files = export_processed_tables(processed_tables, paths["data_processed_dir"])

    summary = {
        "raw_transaction_rows": int(len(casted_datasets["transaction_data"])),
        "clean_transaction_rows": int(len(transactions_cleaned)),
        "removed_transaction_rows": int(len(casted_datasets["transaction_data"]) - len(transactions_cleaned)),
        "valid_households": int(transactions_cleaned["household_key"].nunique()),
        "valid_products": int(transactions_cleaned["PRODUCT_ID"].nunique()),
        "global_fallback_threshold_days": round(global_threshold, 2),
        "churn_distribution": {
            str(label): int(count)
            for label, count in customer_churn["is_churn"].value_counts().sort_index().items()
        },
        "exported_files": exported_files,
    }

    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DDM churn preprocessing pipeline.")
    parser.add_argument("--config", default="config/paths.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preprocessing(args.config)
