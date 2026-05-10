"""
Churn Prediction: Propensity Score Matching (PSM) & Business Impact Analysis
Pipeline for causal inference and ROMI.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import chi2_contingency, chi2
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CORE FUNCTIONS FOR PSM PIPELINE
# ==========================================

def integrate_data(features_path, treatment_path, ps_path):
    features_df = pd.read_csv(features_path)
    treatment_df = pd.read_csv(treatment_path)
    propensity_df = pd.read_csv(ps_path)
    merged_df = features_df.merge(treatment_df, on='household_key', how='inner')
    merged_df = merged_df.merge(propensity_df, on='household_key', how='inner')
    treatment_col = 'is_treated_x' if 'is_treated_x' in merged_df.columns else 'is_treated'
    return merged_df, treatment_col

def perform_matching(df, treatment_col, ps_col='propensity_score', caliper=0.05, random_state=42):
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    # Shuffle treated to avoid order bias (do not re-sort).
    treated = treated.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    treated['pair_id'] = np.nan
    control['pair_id'] = np.nan
    pair_counter = 1
    
    matched_control_indices = set()
    matched_treated_indices = []
    matched_control_actual_indices = []
    
    control_ps = control[ps_col].values
    control_indices = control.index.values
    
    for i, t_row in treated.iterrows():
        t_ps = t_row[ps_col]
        
        available_mask = ~np.isin(control_indices, list(matched_control_indices))
        if not available_mask.any():
            break
            
        available_controls_ps = control_ps[available_mask]
        available_controls_idx = control_indices[available_mask]
        
        distances = np.abs(available_controls_ps - t_ps)
        min_pos = np.argmin(distances)
        min_dist = distances[min_pos]
        
        if min_dist <= caliper:
            matched_treated_indices.append(i)
            matched_control_idx = available_controls_idx[min_pos]
            matched_control_actual_indices.append(matched_control_idx)
            matched_control_indices.add(matched_control_idx)
            
            treated.at[i, 'pair_id'] = pair_counter
            control.at[matched_control_idx, 'pair_id'] = pair_counter
            pair_counter += 1
            
    matched_treated_df = treated.iloc[matched_treated_indices]
    matched_control_df = control.loc[matched_control_actual_indices]
    
    matched_df = pd.concat([matched_treated_df, matched_control_df]).reset_index(drop=True)
    return matched_df, len(matched_treated_indices), len(treated) - len(matched_treated_indices)

def calculate_smd(df, treatment_col, features):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    smds = {}
    for f in features:
        if f not in df.columns:
            continue
        mean_t, mean_c = treated[f].mean(), control[f].mean()
        var_t, var_c = treated[f].var(), control[f].var()
        smd = 0 if var_t + var_c == 0 else np.abs(mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)
        smds[f] = smd
    return smds

def plot_love_plot(smd_before, smd_after):
    smd_df = pd.DataFrame({
        'Feature': list(smd_before.keys()),
        'SMD_Before': list(smd_before.values()),
        'SMD_After': [smd_after.get(f, np.nan) for f in smd_before.keys()]
    }).sort_values(by='SMD_Before', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smd_df['SMD_Before'], y=smd_df['Feature'], mode='markers', name='Before', marker=dict(color='red', size=10)))
    fig.add_trace(go.Scatter(x=smd_df['SMD_After'], y=smd_df['Feature'], mode='markers', name='After', marker=dict(color='blue', size=10)))
    
    for _, row in smd_df.iterrows():
        fig.add_trace(go.Scatter(x=[row['SMD_Before'], row['SMD_After']], y=[row['Feature'], row['Feature']], mode='lines', line=dict(color='gray', dash='dot'), showlegend=False, hoverinfo='skip'))
        
    fig.add_shape(type="line", x0=0.1, y0=-0.5, x1=0.1, y1=len(smd_df)-0.5, line=dict(color="black", dash="dash"))
    fig.add_shape(type="line", x0=0, y0=-0.5, x1=0, y1=len(smd_df)-0.5, line=dict(color="black", width=1))
    fig.update_layout(title='Love Plot: Standardized Mean Differences', height=800, template='plotly_white')
    return fig

def plot_ps_distribution(df, treatment_col, ps_col='propensity_score'):
    treated_ps = df[df[treatment_col] == 1][ps_col].values
    control_ps = df[df[treatment_col] == 0][ps_col].values
    try:
        fig = ff.create_distplot([treated_ps, control_ps], ['Treated', 'Control'], colors=['red', 'blue'], show_hist=False, show_rug=False)
        fig.update_layout(title='Propensity Score Distribution After Matching', template='plotly_white')
        return fig
    except:
        return None

def analyze_outcome(df_matched, treatment_col, silent=False):
    churn_stats = df_matched.groupby(treatment_col)['churn_flag'].agg(['mean', 'sum', 'count'])
    churn_stats.columns = ['Churn_Rate', 'Churn_Count', 'Total_Count']
    p0 = churn_stats.loc[0, 'Churn_Rate'] if 0 in churn_stats.index else 0
    p1 = churn_stats.loc[1, 'Churn_Rate'] if 1 in churn_stats.index else 0
    
    ate = p0 - p1
    relative_lift = ate / p0 if p0 > 0 else 0
    
    if len(df_matched) > 0 and len(df_matched[treatment_col].unique()) > 1:
        contingency_table = pd.crosstab(df_matched[treatment_col], df_matched['churn_flag'])
        chi2, p_val, dof, ex = chi2_contingency(contingency_table)
    else:
        p_val = 1.0
        
    if not silent:
        print("=== ATE IMPACT ANALYSIS RESULTS ===")
        print(f"Control churn rate (matched): {p0:.2%}")
        print(f"Treated churn rate (matched): {p1:.2%}")
        if ate >= 0:
            print(f"Observed effect (ATE): Decrease of {ate:.2%} in churn")
            print(f"Relative lift: {relative_lift:.2%}")
        else:
            print(f"Observed effect (ATE): Increase of {-ate:.2%} in churn")
            print(f"Relative lift: {-relative_lift:.2%}")
        print(f"Chi-square p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            if ate > 0:
                print("\n=> CHI-SQUARE CONCLUSION: The coupon reduces churn (statistically significant).")
            else:
                print("\n=> CHI-SQUARE CONCLUSION: The coupon increases churn (statistically significant) - counterproductive.")
        else:
            print("\n=> CHI-SQUARE CONCLUSION: Not enough statistical evidence.")
            
        if 'pair_id' in df_matched.columns:
            t_df = df_matched[df_matched[treatment_col] == 1][['pair_id', 'churn_flag']].rename(columns={'churn_flag': 'churn_t'})
            c_df = df_matched[df_matched[treatment_col] == 0][['pair_id', 'churn_flag']].rename(columns={'churn_flag': 'churn_c'})
            pair_df = t_df.merge(c_df, on='pair_id', how='inner')
            b = len(pair_df[(pair_df['churn_c'] == 0) & (pair_df['churn_t'] == 1)])
            c_ = len(pair_df[(pair_df['churn_c'] == 1) & (pair_df['churn_t'] == 0)])
            if b + c_ > 0:
                mcnemar_chi2 = (abs(b - c_) - 1)**2 / (b + c_)
                mcnemar_pval = 1 - chi2.cdf(mcnemar_chi2, 1)
            else:
                mcnemar_chi2 = 0
                mcnemar_pval = 1.0
            print(f"\nMcNemar test p-value (matched pairs): {mcnemar_pval:.4f}")
            if mcnemar_pval < 0.05:
                if ate > 0:
                    print("=> McNemar CONCLUSION: The coupon DOES reduce churn (statistically significant on pairs).")
                else:
                    print("=> McNemar CONCLUSION: The coupon DOES increase churn (statistically significant on pairs).")
            else:
                print("=> McNemar CONCLUSION: Not enough statistical evidence on matched pairs.")
            
    return ate, p0, p1

def bootstrap_ate_diff(target_matched, nontarget_matched, treatment_col, n_bootstrap=1000, random_state=42):
    np.random.seed(random_state)
    diffs = []
    
    for _ in range(n_bootstrap):
        t_target = target_matched.groupby(treatment_col, group_keys=False).apply(lambda x: x.sample(frac=1, replace=True))
        t_nontarget = nontarget_matched.groupby(treatment_col, group_keys=False).apply(lambda x: x.sample(frac=1, replace=True))
        
        p0_t = t_target[t_target[treatment_col] == 0]['churn_flag'].mean()
        p1_t = t_target[t_target[treatment_col] == 1]['churn_flag'].mean()
        ate_t = p0_t - p1_t
        
        p0_nt = t_nontarget[t_nontarget[treatment_col] == 0]['churn_flag'].mean()
        p1_nt = t_nontarget[t_nontarget[treatment_col] == 1]['churn_flag'].mean()
        ate_nt = p0_nt - p1_nt
        
        diffs.append(ate_t - ate_nt)
        
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    
    return np.mean(diffs), ci_lower, ci_upper

def run_sensitivity_analysis(target_df, treatment_col, ate_target):
    treated_target_count = len(target_df[target_df[treatment_col] == 1])
    
    if 'predicted_discounted_value_60d_if_active' not in target_df.columns:
        raise ValueError("Missing the predicted_discounted_value_60d_if_active column required to calculate CLV.")
    avg_clv_target = target_df['predicted_discounted_value_60d_if_active'].mean()
    
    print("\nNote: The average CLV is extracted from Member 5 (M5)'s predictive model and is based on historical behavior.\n")
    
    scenarios = [
        {'Scenario': 'Conservative', 'Cost': 5.0, 'Margin_Ratio': 0.20},
        {'Scenario': 'Moderate', 'Cost': 2.5, 'Margin_Ratio': 0.25},
        {'Scenario': 'Optimistic', 'Cost': 1.5, 'Margin_Ratio': 0.30}
    ]
    
    results = []
    for sc in scenarios:
        cost = sc['Cost']
        margin_ratio = sc['Margin_Ratio']
        clv = avg_clv_target * margin_ratio
        benefits = treated_target_count * ate_target * clv
        total_costs = treated_target_count * cost
        roi = (benefits - total_costs) / total_costs if total_costs > 0 else 0
        
        results.append({
            'Scenario': sc['Scenario'],
            'Cost/Coupon ($)': cost,
            'Margin Ratio': f"{margin_ratio:.0%}",
            'Benefit ($)': round(benefits, 2),
            'Total Cost ($)': round(total_costs, 2),
            'ROI': f"{roi:.2%}"
        })
    return pd.DataFrame(results)

# ==========================================
# MAIN PIPELINE
# ==========================================

if __name__ == "__main__":
    # Data paths (adjust to the actual folder structure).
    features_path = '../models/final_ML_features.csv'
    treatment_path = '../models/psm_inputs/psm_treatment_flags.csv'
    ps_path = '../models/m6_handoff/propensity_scores_for_psm.csv'
    
    # Steps 1 & 2: data integration and Common Support filtering.
    df_merged, treatment_col = integrate_data(features_path, treatment_path, ps_path)
    print(f"Total rows after merging: {len(df_merged)}")
    
    if 'common_support_flag' in df_merged.columns:
        df_cs = df_merged[df_merged['common_support_flag'] == True].copy()
        print(f"Number of samples removed for being outside Common Support: {len(df_merged) - len(df_cs)}")
    else:
        df_cs = df_merged.copy()
    
    # Step 3: interaction test (logistic regression).
    formula = f"churn_flag ~ {treatment_col} * risk_decile"
    try:
        model = smf.logit(formula=formula, data=df_cs).fit(disp=0)
        print("\n=== INTERACTION TEST (Logistic Regression) ===")
        print(model.summary().tables[1])
        interaction_pvalue = model.pvalues.get(f"{treatment_col}:risk_decile", 1.0)
        print(f"\nP-value of the interaction term (Treatment * Risk Decile): {interaction_pvalue:.4f}")
        if interaction_pvalue < 0.05:
            print("=> The interaction IS statistically significant. We have strong evidence to analyze subgroups by decile.")
        else:
            print("=> The interaction is NOT clearly statistically significant. There is no strong evidence of different effects across deciles. However, based on the business hypothesis, we still examine subgroup [3,5,6,7].")
    except Exception as e:
        print("Error fitting interaction model:", e)
    
    # Step 4: subgroup re-matching.
    target_deciles = [3, 5, 6, 7]
    df_target_raw = df_cs[df_cs['risk_decile'].isin(target_deciles)].copy()
    df_nontarget_raw = df_cs[~df_cs['risk_decile'].isin(target_deciles)].copy()
    
    print("\n--- SUBGROUP MATCHING ---")
    print(f"Initial sample count in Target: {len(df_target_raw)} (Treated: {df_target_raw[treatment_col].sum()})")
    target_matched, num_matches_tgt, discarded_tgt = perform_matching(df_target_raw, treatment_col, caliper=0.05)
    print(f"After matching Target: {len(target_matched)} samples ({num_matches_tgt} pairs). Dropped: {discarded_tgt} treated.")
    
    print(f"\nInitial sample count in Non-Target: {len(df_nontarget_raw)} (Treated: {df_nontarget_raw[treatment_col].sum()})")
    nontarget_matched, num_matches_ntgt, discarded_ntgt = perform_matching(df_nontarget_raw, treatment_col, caliper=0.05)
    print(f"After matching Non-Target: {len(nontarget_matched)} samples ({num_matches_ntgt} pairs). Dropped: {discarded_ntgt} treated.")
    
    # Check SMD for the Target group after re-matching.
    features_subgroup = [
        'Frequency', 'Monetary', 'Basket_Value_Std', 'customer_lifetime', 
        'Recency_Capped', 'Inactive_Days_Ratio', 'active_weeks_ratio', 
        'Avg_Items_Per_Basket', 'IPT_mean', 'IPT_std', 'IPT_CV', 
        'Private_Brand_Ratio', 'Rolling_Freq_Slope', 'Days_Since_Last_Camp',
        'has_demographic_info'
    ]
    features_subgroup = [f for f in features_subgroup if f in target_matched.columns]
    
    smd_target = calculate_smd(target_matched, treatment_col, features_subgroup)
    smd_target_df = pd.DataFrame({
        'Feature': list(smd_target.keys()),
        'SMD': list(smd_target.values())
    }).sort_values(by='SMD', ascending=False)
    
    print("\n=== BALANCE CHECK (SMD) FOR TARGET GROUP - ROUND 2 ===")
    print(smd_target_df.to_string(index=False))
    
    is_balanced = all(smd_target_df['SMD'] < 0.1)
    if not is_balanced:
        print("\n[WARNING] Some variables still have SMD > 0.1. The ATE result may be biased. There is not enough evidence to conclude ATE robustly.")
    else:
        print("\n=> All variables have SMD < 0.1. The data are fully balanced.")
    
    if len(target_matched) < 60:
        print("\n[WARNING] The matched sample size is still too small (<30 per group). Do NOT conclude ATE for this group because statistical power is very weak.")
    
    # Step 5: ATE and bootstrap.
    print("\n--- ATE FOR TARGET GROUP ---")
    if is_balanced and len(target_matched) >= 60:
        ate_t, _, _ = analyze_outcome(target_matched, treatment_col, silent=False)
    else:
        print("Skipping Target ATE analysis because the data are unbalanced or the sample size is too small.")
        ate_t = 0
    
    print("\n--- ATE FOR NON-TARGET GROUP ---")
    ate_nt, _, _ = analyze_outcome(nontarget_matched, treatment_col, silent=False)
    
    if is_balanced and len(target_matched) >= 60:
        print("\n--- BOOTSTRAP TEST FOR ATE DIFFERENCE (1000 iterations) ---")
        mean_diff, ci_lower, ci_upper = bootstrap_ate_diff(target_matched, nontarget_matched, treatment_col, n_bootstrap=1000)
        print(f"Delta ATE (Target - NonTarget): {mean_diff:.2%}")
        print(f"95% CI: [{ci_lower:.2%}, {ci_upper:.2%}]")
        if ci_lower > 0 or ci_upper < 0:
            print("=> The confidence interval does NOT include 0. There is a statistically significant difference in coupon effectiveness between Target and Non-Target.")
        else:
            print("=> The confidence interval includes 0. There is not enough evidence to claim that this difference is fully driven by segment characteristics.")
    else:
        print("\n[WARNING] Bootstrap ATE comparison cannot be performed because the Target group is not balanced (SMD > 0.1) or the sample size is too small.\nThe Non-Target result is for reference only.")
    
    # Step 6: financial simulation.
    if is_balanced and len(target_matched) >= 60 and ate_t > 0:
        print("\n=== SENSITIVITY ANALYSIS (ROMI) ===")
        romi_df = run_sensitivity_analysis(target_matched, treatment_col, ate_t)
        print(romi_df.to_string(index=False))
    else:
        print("\nThe conditions ATE > 0, sufficient sample size, or balance are not satisfied. Skipping financial analysis.")