import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PROCESSED = os.path.join(BASE_DIR, 'Data', 'Processed')
DATA_RAW = os.path.join(BASE_DIR, 'Data', 'Raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_and_filter_data():
    """Loads raw/processed datasets and applies strict CUT-OFF filters."""
    print("1. Loading datasets...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    transactions = pd.read_parquet(os.path.join(DATA_PROCESSED, 'transactions_master.parquet'), engine='pyarrow')
    customer = pd.read_parquet(os.path.join(DATA_PROCESSED, 'customer_base_labeled.parquet'), engine='pyarrow')
    demographics = pd.read_parquet(os.path.join(DATA_PROCESSED, 'demographics_imputed.parquet'), engine='pyarrow')
    product = pd.read_parquet(os.path.join(DATA_RAW, 'product.parquet'), engine='pyarrow')
    causal_data = pd.read_parquet(os.path.join(DATA_RAW, 'causal_data.parquet'), engine='pyarrow')
    campaign_table = pd.read_parquet(os.path.join(DATA_PROCESSED, 'campaign_table_clean.parquet'), engine='pyarrow')
    campaign_desc = pd.read_parquet(os.path.join(DATA_PROCESSED, 'campaign_desc_clean.parquet'), engine='pyarrow')

    # AIRTIGHT CUT-OFF
    CUT_OFF_DAY = 711 - 60 
    print(f"   -> Cut-off Date set to Day {CUT_OFF_DAY}")

    # Transaction Filter (Strictly <)
    obs_txns = transactions[transactions['DAY'] < CUT_OFF_DAY].copy()
    obs_txns['household_key'] = obs_txns['household_key'].astype(str)

    # Causal Filter (Week resolution)
    cutoff_week = int(np.floor(CUT_OFF_DAY / 7))
    causal_obs = causal_data[causal_data['WEEK_NO'] <= cutoff_week].copy()

    # Initialize Base DataFrame
    df_final = customer[['household_key', 'is_churn']].copy()
    df_final.rename(columns={'is_churn': 'churn_flag'}, inplace=True)
    df_final['household_key'] = df_final['household_key'].astype(str)

    return obs_txns, causal_obs, demographics, product, campaign_table, campaign_desc, df_final, CUT_OFF_DAY

def build_rfm_and_behavior(obs_txns, CUT_OFF_DAY):
    """Engineers RFM, Inter-Purchase Time (IPT), and Basket characteristics."""
    print("2. Engineering RFM, Stability (IPT), and Behavior...")
    
    # Base aggregations
    agg_funcs = {
        'BASKET_ID': ['nunique', 'count'], 
        'SALES_VALUE': ['sum', 'std'],     
        'COUPON_DISC': lambda x: np.sum(np.abs(x)), 
        'DAY': ['min', 'max']  
    }
    cust_feats = obs_txns.groupby('household_key').agg(agg_funcs).reset_index()
    cust_feats.columns = ['household_key', 'Frequency', 'Total_Items', 'Monetary', 'Basket_Value_Std', 'Total_Coupon_Discount', 'first_day', 'last_day']

    # Lifetime & Capped Recency
    cust_feats['customer_lifetime'] = CUT_OFF_DAY - cust_feats['first_day']
    cust_feats['customer_lifetime'] = np.where(cust_feats['customer_lifetime'] <= 0, 1, cust_feats['customer_lifetime'])
    
    raw_recency = CUT_OFF_DAY - cust_feats['last_day']
    cust_feats['Recency_Capped'] = np.clip(raw_recency, 0, 90)
    cust_feats['Inactive_Days_Ratio'] = np.clip(raw_recency / cust_feats['customer_lifetime'], 0, 1)

    # Active Weeks Ratio
    active_weeks = obs_txns.groupby('household_key')['WEEK_NO'].nunique().reset_index(name='Active_Weeks')
    cust_feats = pd.merge(cust_feats, active_weeks, on='household_key', how='left')
    cust_feats['active_weeks_ratio'] = np.clip(cust_feats['Active_Weeks'] / ((cust_feats['customer_lifetime'] / 7) + 1), 0, 1)

    # Promo Usage Ratio
    txns_with_coupon = obs_txns[obs_txns['COUPON_DISC'] < 0].groupby('household_key')['BASKET_ID'].nunique().reset_index(name='Trips_With_Coupon')
    cust_feats = pd.merge(cust_feats, txns_with_coupon, on='household_key', how='left').fillna(0)
    cust_feats['promo_usage_ratio'] = cust_feats['Trips_With_Coupon'] / (cust_feats['Frequency'] + 1e-5)

    # Basket Metrics
    cust_feats['Avg_Items_Per_Basket'] = cust_feats['Total_Items'] / (cust_feats['Frequency'] + 1e-5)
    cust_feats['coupon_dependency'] = cust_feats['Total_Coupon_Discount'] / (cust_feats['Monetary'] + 1e-5)
    
    # Cleanup
    cust_feats.drop(columns=['Total_Items', 'Total_Coupon_Discount', 'first_day', 'last_day', 'Active_Weeks', 'Trips_With_Coupon'], inplace=True)
    cust_feats.fillna(0, inplace=True)

    # Habit & Stability (IPT)
    unique_days = obs_txns[['household_key', 'DAY']].drop_duplicates().sort_values(['household_key', 'DAY'])
    unique_days['Prev_DAY'] = unique_days.groupby('household_key')['DAY'].shift(1)
    unique_days['IPT'] = unique_days['DAY'] - unique_days['Prev_DAY']

    ipt_stats = unique_days.groupby('household_key').agg(IPT_mean=('IPT', 'mean'), IPT_std=('IPT', 'std')).reset_index()
    
    # Fix perfect stability logic
    ipt_stats['IPT_std'] = ipt_stats['IPT_std'].fillna(ipt_stats['IPT_mean'])
    cust_life_map = dict(zip(cust_feats['household_key'], cust_feats['customer_lifetime']))
    ipt_stats['IPT_mean'] = ipt_stats['IPT_mean'].fillna(ipt_stats['household_key'].map(cust_life_map))
    ipt_stats['IPT_std'] = ipt_stats['IPT_std'].fillna(ipt_stats['household_key'].map(cust_life_map))
    ipt_stats['IPT_CV'] = ipt_stats['IPT_std'] / (ipt_stats['IPT_mean'] + 1e-5)
    ipt_stats.fillna(0, inplace=True)

    return pd.merge(cust_feats, ipt_stats, on='household_key', how='left')

def build_marketing_and_trend(obs_txns, causal_obs, product, CUT_OFF_DAY):
    """Engineers Brand Affinity, Point-of-Sale Responsiveness, Rolling Freq Trend, and Primary Store."""
    print("3. Engineering Brand Affinity, Causal, Rolling Trend, and Primary Store...")
    
    # Brand Affinity (Laplace)
    if 'BRAND' not in obs_txns.columns:
        obs_txns = pd.merge(obs_txns, product[['PRODUCT_ID', 'BRAND']], on='PRODUCT_ID', how='left')
    brand_qty = obs_txns.groupby(['household_key', 'BRAND'])['QUANTITY'].sum().unstack(fill_value=0).reset_index()
    brand_qty['Private_Brand_Ratio'] = (brand_qty.get('Private', 0) + 1) / (brand_qty.get('Private', 0) + brand_qty.get('National', 0) + 2)

    # Causal Data Responsiveness
    causal_obs['PRODUCT_ID'] = causal_obs['PRODUCT_ID'].astype(str)
    causal_obs['WEEK_NO'] = causal_obs['WEEK_NO'].astype(int)
    obs_txns['PRODUCT_ID'] = obs_txns['PRODUCT_ID'].astype(str)
    obs_txns['WEEK_NO'] = obs_txns['WEEK_NO'].astype(int)

    causal_obs['is_display'] = np.where(~causal_obs['display'].isin(['0', ' ', 'None', 'NaN']), 1, 0)
    causal_obs['is_mailer'] = np.where(~causal_obs['mailer'].isin(['0', ' ', 'None', 'NaN']), 1, 0)
    causal_agg = causal_obs.groupby(['WEEK_NO', 'PRODUCT_ID']).agg({'is_display': 'max', 'is_mailer': 'max'}).reset_index()

    txn_causal = pd.merge(obs_txns[['household_key', 'WEEK_NO', 'PRODUCT_ID', 'QUANTITY']], causal_agg, on=['WEEK_NO', 'PRODUCT_ID'], how='left').fillna(0)
    txn_causal['Qty_Display'] = txn_causal['QUANTITY'] * txn_causal['is_display']
    txn_causal['Qty_Mailer'] = txn_causal['QUANTITY'] * txn_causal['is_mailer']

    causal_features = txn_causal.groupby('household_key').agg(
        Total_Qty=('QUANTITY', 'sum'),
        Qty_Display=('Qty_Display', 'sum'),
        Qty_Mailer=('Qty_Mailer', 'sum')
    ).reset_index()

    causal_features['Display_Responsiveness'] = causal_features['Qty_Display'] / (causal_features['Total_Qty'] + 1e-5)
    causal_features['Mailer_Responsiveness'] = causal_features['Qty_Mailer'] / (causal_features['Total_Qty'] + 1e-5)
    causal_features.drop(columns=['Total_Qty', 'Qty_Display', 'Qty_Mailer'], inplace=True)

    # Rolling Trend (Polyfit over 3 months)
    obs_txns['days_to_cutoff'] = CUT_OFF_DAY - obs_txns['DAY']
    m1 = obs_txns[obs_txns['days_to_cutoff'] <= 30].groupby('household_key')['BASKET_ID'].nunique().reset_index(name='M1')
    m2 = obs_txns[(obs_txns['days_to_cutoff'] > 30) & (obs_txns['days_to_cutoff'] <= 60)].groupby('household_key')['BASKET_ID'].nunique().reset_index(name='M2')
    m3 = obs_txns[(obs_txns['days_to_cutoff'] > 60) & (obs_txns['days_to_cutoff'] <= 90)].groupby('household_key')['BASKET_ID'].nunique().reset_index(name='M3')

    trend_df = pd.DataFrame({'household_key': obs_txns['household_key'].unique()})
    trend_df = trend_df.merge(m1, on='household_key', how='left').merge(m2, on='household_key', how='left').merge(m3, on='household_key', how='left').fillna(0)
    
    def calc_slope(row):
        return np.polyfit([1, 2, 3], [row['M3'], row['M2'], row['M1']], 1)[0]
    
    trend_df['Rolling_Freq_Slope'] = trend_df.apply(calc_slope, axis=1)

    # Primary Store Affinity
    primary_store = obs_txns.groupby('household_key')['STORE_ID'].agg(
        lambda x: pd.Series.mode(x)[0] if not x.mode().empty else np.nan
    ).reset_index(name='Primary_Store_ID')
    primary_store['Primary_Store_ID'] = primary_store['Primary_Store_ID'].astype(str)

    return brand_qty[['household_key', 'Private_Brand_Ratio']], causal_features, trend_df[['household_key', 'Rolling_Freq_Slope']], primary_store

def build_campaign_and_demo(campaign_table, campaign_desc, demographics, obs_txns, CUT_OFF_DAY):
    """Engineers Campaign Exposure, Fatigue, Categorization, and Demographic Proxies."""
    print("4. Engineering Campaign & Demographics...")
    
    # Combine Campaign Data
    if 'DESCRIPTION' in campaign_table.columns:
        campaign_table = campaign_table.drop(columns=['DESCRIPTION'])
    camp_full = pd.merge(campaign_table, campaign_desc, on='CAMPAIGN', how='left')

    camp_obs = camp_full[camp_full['START_DAY'] < CUT_OFF_DAY].copy()
    camp_obs['household_key'] = camp_obs['household_key'].astype(str)

    # Campaign Categories
    camp_types = pd.crosstab(camp_obs['household_key'], camp_obs['DESCRIPTION']).add_prefix('Camp_Count_').reset_index()
    for col in ['Camp_Count_TypeA', 'Camp_Count_TypeB', 'Camp_Count_TypeC']:
        if col not in camp_types.columns:
            camp_types[col] = 0
    camp_types['Total_Campaigns_Received'] = camp_types[['Camp_Count_TypeA', 'Camp_Count_TypeB', 'Camp_Count_TypeC']].sum(axis=1)

    # Campaign Fatigue (Last 30D)
    camp_obs['days_to_cutoff_camp'] = CUT_OFF_DAY - camp_obs['START_DAY']
    fatigue = camp_obs[camp_obs['days_to_cutoff_camp'] <= 30].groupby('household_key').size().reset_index(name='Campaigns_Last_30D')

    # Campaign Recency
    camp_recency = camp_obs.groupby('household_key')['START_DAY'].max().reset_index()
    camp_recency['Days_Since_Last_Camp'] = CUT_OFF_DAY - camp_recency['START_DAY']

    campaign_features = pd.merge(camp_types, fatigue, on='household_key', how='outer').fillna(0)
    campaign_features = pd.merge(campaign_features, camp_recency[['household_key', 'Days_Since_Last_Camp']], on='household_key', how='outer')

    # Demographics
    demographics['household_key'] = demographics['household_key'].astype(str)
    demo_feature = pd.DataFrame({'household_key': obs_txns['household_key'].unique()})
    demo_feature = pd.merge(demo_feature, demographics[['household_key', 'AGE_DESC']], on='household_key', how='left')
    demo_feature['has_demographic_info'] = np.where(demo_feature['AGE_DESC'].notna(), 1, 0)
    
    return campaign_features, demo_feature[['household_key', 'has_demographic_info']]

def assemble_and_export(df_final, cust_feats, brand_feats, causal_feats, trend_df, primary_store, camp_feats, demo_feat, CUT_OFF_DAY):
    """Merges features, splits data, scales, applies SMOTE, and exports outputs."""
    print("5. Assembling Data, Scaling, and Applying SMOTETomek...")
    
    # Merge all dataframes
    df_final = pd.merge(df_final, cust_feats, on='household_key', how='inner')
    df_final = pd.merge(df_final, brand_feats, on='household_key', how='left')
    df_final = pd.merge(df_final, causal_feats, on='household_key', how='left')
    df_final = pd.merge(df_final, trend_df, on='household_key', how='left')
    df_final = pd.merge(df_final, primary_store, on='household_key', how='left')
    df_final = pd.merge(df_final, camp_feats, on='household_key', how='left')
    df_final = pd.merge(df_final, demo_feat, on='household_key', how='left')

    # Final imputation
    df_final['Days_Since_Last_Camp'].fillna(CUT_OFF_DAY, inplace=True)
    df_final['Primary_Store_ID'].fillna('Unknown', inplace=True)
    df_final.fillna(0, inplace=True)

    print(f"   -> Final DataFrame shape (Before SMOTE): {df_final.shape}")
    df_final.to_csv(os.path.join(MODELS_DIR, 'final_ML_features.csv'), index=False)

    # ML Pipeline Splitting
    keys = df_final['household_key']
    stores = df_final['Primary_Store_ID']
    # Drop categorical features and IDs before scaling
    X = df_final.drop(columns=['household_key', 'churn_flag', 'Primary_Store_ID'])
    y = df_final['churn_flag']
    
    X_train, X_test, y_train, y_test, keys_train, keys_test, stores_train, stores_test = train_test_split(
        X, y, keys, stores, test_size=0.2, random_state=42, stratify=y
    )

    # Airtight Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Handle Imbalance
    smt = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train_scaled, y_train)

    print("*" * 60)
    print("NOTICE FOR MODELING TEAM (M5/M6):")
    print("1. Train set has been processed with SMOTETomek.")
    print("2. 'Primary_Store_ID' is categorical and re-attached to the files.")
    print("   Apply Target Encoding using train set distributions if using Linear/NN models.")
    print("*" * 60)

    # Pack DataFrames
    train_final_df = pd.concat([X_train_resampled, y_train_resampled.reset_index(drop=True)], axis=1)
    test_final_df = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    
    # Restore ID and Store ID for Test set
    test_final_df['household_key'] = keys_test.values
    test_final_df['Primary_Store_ID'] = stores_test.values

    # Restore ID and Store ID for Train set 
    # (Synthetic rows generated by SMOTE will be given a placeholder tag)
    train_store_info = pd.DataFrame({
        'household_key': keys_train.values,
        'Primary_Store_ID': stores_train.values
    })
    train_final_df = pd.concat([train_final_df, train_store_info], axis=1)
    train_final_df['household_key'].fillna('Synthetic', inplace=True)
    train_final_df['Primary_Store_ID'].fillna('Synthetic', inplace=True)

    # Export
    train_final_df.to_parquet(os.path.join(MODELS_DIR, 'final_train_features.parquet'), index=False)
    test_final_df.to_parquet(os.path.join(MODELS_DIR, 'final_test_features.parquet'), index=False)
    
    print(f"\n[SUCCESS] Pipeline Completed! Files exported to '{MODELS_DIR}' directory.")

def run_pipeline():
    obs_txns, causal_obs, demographics, product, campaign_table, campaign_desc, df_final, CUT_OFF_DAY = load_and_filter_data()
    
    cust_feats = build_rfm_and_behavior(obs_txns, CUT_OFF_DAY)
    brand_feats, causal_feats, trend_df, primary_store = build_marketing_and_trend(obs_txns, causal_obs, product, CUT_OFF_DAY)
    camp_feats, demo_feat = build_campaign_and_demo(campaign_table, campaign_desc, demographics, obs_txns, CUT_OFF_DAY)
    
    assemble_and_export(df_final, cust_feats, brand_feats, causal_feats, trend_df, primary_store, camp_feats, demo_feat, CUT_OFF_DAY)

if __name__ == "__main__":
    run_pipeline()