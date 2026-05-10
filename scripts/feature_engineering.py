import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PROCESSED = os.path.join(BASE_DIR, 'Data', 'Processed')
DATA_RAW = os.path.join(BASE_DIR, 'Data', 'Raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PSM_DIR = os.path.join(MODELS_DIR, 'psm_inputs')

def load_and_filter_data():
    """Loads raw/processed datasets and applies strict CUT-OFF filters."""
    print("1. Loading datasets...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PSM_DIR, exist_ok=True)
    
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

def assemble_final_dataset(customer_labels, cust_feats, brand_qty, causal_features, 
                           trend_df, campaign_features, demo_feature, primary_store, CUT_OFF_DAY):
    """Merges all engineered components into a single comprehensive DataFrame."""
    print("\n--- STEP 5: ASSEMBLING FINAL DATASET ---")
    
    # Start with base labels
    df_final = customer_labels[['household_key', 'churn_flag']].copy()
    
    # Merge all feature tables
    df_final = pd.merge(df_final, cust_feats, on='household_key', how='inner')
    df_final = pd.merge(df_final, brand_qty[['household_key', 'Private_Brand_Ratio']], on='household_key', how='left')
    df_final = pd.merge(df_final, causal_features, on='household_key', how='left')
    df_final = pd.merge(df_final, trend_df, on='household_key', how='left')
    df_final = pd.merge(df_final, campaign_features, on='household_key', how='left')
    df_final = pd.merge(df_final, demo_feature, on='household_key', how='left')
    df_final = pd.merge(df_final, primary_store, on='household_key', how='left')

    # Handle NaN values globally
    if 'Days_Since_Last_Camp' in df_final.columns:
        df_final['Days_Since_Last_Camp'].fillna(CUT_OFF_DAY, inplace=True)
    if 'Primary_Store_ID' in df_final.columns:
        df_final['Primary_Store_ID'].fillna('Unknown', inplace=True)
        
    df_final.fillna(0, inplace=True)
    print(f"Master DataFrame assembled. Shape: {df_final.shape}")
    
    return df_final

def generate_psm_flags(df_final, CUT_OFF_DAY):
    """Creates Method C (Intensity-based) treatment flags for M6 Causal Inference."""
    print("\n--- STEP 6: GENERATING TREATMENT FLAGS FOR PSM (METHOD C) ---")
    
    psm_flags = pd.DataFrame({'household_key': df_final['household_key'].unique()})
    
    # Extract promo_usage_ratio
    if 'promo_usage_ratio' in df_final.columns:
        psm_flags = pd.merge(psm_flags, df_final[['household_key', 'promo_usage_ratio']], on='household_key', how='left')
    else:
        psm_flags['promo_usage_ratio'] = 0 
        
    psm_flags['promo_usage_ratio'].fillna(0, inplace=True)
    mean_promo_ratio = psm_flags['promo_usage_ratio'].mean()
    print(f"-> Threshold (Mean Promo Ratio): {mean_promo_ratio:.4f}")

    # Assign treatment flag (1 = Heavy User, 0 = Light/No User)
    psm_flags['is_treated'] = (psm_flags['promo_usage_ratio'] > mean_promo_ratio).astype(int)
    
    # Format treatment source
    psm_flags['treatment_source'] = psm_flags['is_treated'].apply(
        lambda x: 'Heavy_Promo_User' if x == 1 else 'Light/No_Promo_User'
    )
    psm_flags['treatment_cutoff_day'] = CUT_OFF_DAY
    
    # Export PSM file
    psm_export_df = psm_flags[['household_key', 'is_treated', 'treatment_source', 'treatment_cutoff_day']]
    psm_csv_path = os.path.join(PSM_DIR, 'psm_treatment_flags.csv')
    psm_export_df.to_csv(psm_csv_path, index=False)
    
    print(f"Exported {len(psm_export_df)} PSM flags to: {psm_csv_path}")

def export_multi_version_features(df_final):
    """Exports independent datasets tailored for Tree-based and Linear models."""
    print("\n--- STEP 8: EXPORTING MULTI-VERSION FEATURES ---")
    
    # Define structurally flawed features causing perfect multicollinearity
    linear_drop_cols = [
        'Total_Campaigns_Received',  # VIF = inf
        'IPT_mean',                  # Highly correlated (0.95) with IPT_std
        'IPT_std'                    # We retain IPT_CV to safely represent both
    ]

    # Create independent DataFrames
    df_tree = df_final.copy()
    df_linear = df_final.drop(columns=linear_drop_cols, errors='ignore')

    # Export to CSV (Raw, unscaled versions - M5 handles scaling natively)
    tree_path = os.path.join(MODELS_DIR, 'final_ML_features_tree.csv')
    linear_path = os.path.join(MODELS_DIR, 'final_ML_features_linear.csv')

    df_tree.to_csv(tree_path, index=False)
    df_linear.to_csv(linear_path, index=False)

    print(f"Version A (Tree-based) saved: {tree_path} | Shape: {df_tree.shape}")
    print(f"Version B (Linear) saved:     {linear_path} | Shape: {df_linear.shape}")
    
def run_pipeline():
    obs_txns, causal_obs, demographics, product, campaign_table, campaign_desc, customer_labels, CUT_OFF_DAY = load_and_filter_data()
    
    cust_feats = build_rfm_and_behavior(obs_txns, CUT_OFF_DAY)
    brand_feats, causal_feats, trend_df, primary_store = build_marketing_and_trend(obs_txns, causal_obs, product, CUT_OFF_DAY)
    camp_feats, demo_feat = build_campaign_and_demo(campaign_table, campaign_desc, demographics, obs_txns, CUT_OFF_DAY)
    
    df_final = assemble_final_dataset(
        customer_labels=customer_labels, 
        cust_feats=cust_feats, 
        brand_qty=brand_feats, 
        causal_features=causal_feats, 
        trend_df=trend_df, 
        campaign_features=camp_feats, 
        demo_feature=demo_feat, 
        primary_store=primary_store, 
        CUT_OFF_DAY=CUT_OFF_DAY
    )
    
    generate_psm_flags(df_final, CUT_OFF_DAY)
    export_multi_version_features(df_final)

if __name__ == "__main__":
    run_pipeline()