import pandas as pd
import numpy as np
from pipeline import load_google_ads, load_meta_ads, load_microsoft_ads, aggregate_to_weekly
from feature_engineering import compute_features
from models import BayesianForecaster, SeasonalNaiveForecaster

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true > 0
    if not np.any(mask): return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# FIX P0: Features that leak current-row information and must be replaced
# with last-training-row values when predicting on the test set
LEAKED_FEATURES = [
    'rolling_4w_avg_revenue', 'rolling_4w_std_revenue', 'rolling_8w_avg_revenue',
    'rolling_4w_avg_ROAS', 'rolling_4w_avg_CVR', 'rolling_4w_avg_CPC',
    'efficiency_index', 'yoy_revenue_growth', 'spend_momentum',
    'spend_share'
]

def patch_test_features(test_df, train_df):
    """Replace potentially leaked features in test_df with last training row values.
    
    This simulates what main.py does during production: at prediction time, 
    we only know the last observed values of rolling/efficiency features.
    """
    test_df = test_df.copy()
    last_train_row = train_df.iloc[-1]
    
    for feat in LEAKED_FEATURES:
        if feat in test_df.columns and feat in train_df.columns:
            test_df[feat] = last_train_row[feat]
    
    return test_df

def run_evaluation():
    df_google = load_google_ads('data/google_ads_campaign_stats.csv')
    df_meta = load_meta_ads('data/meta_ads_campaign_stats.csv')
    df_ms = load_microsoft_ads('data/bing_campaign_stats.csv')
    df_all = pd.concat([df_google, df_meta, df_ms], ignore_index=True)
    
    weekly_df = aggregate_to_weekly(df_all)
    features_df = compute_features(weekly_df)
    
    segments = features_df.groupby(['channel', 'campaign_type'])
    
    # Segment-level metrics
    all_actuals = []
    all_p50s = []
    all_p10s = []
    all_p90s = []
    
    # Channel-level roll-up metrics (Stage 2)
    channel_fold_data = {}
    
    for (ch, ct), grp in segments:
        grp = grp.sort_values('week_start').reset_index(drop=True)
        n_weeks = len(grp)
        if n_weeks <= 30:
            continue
        
        # Skip ultra-low-volume segments (mean weekly revenue < $50)
        mean_weekly_rev = grp['conversion_value'].mean()
        if mean_weekly_rev < 50:
            continue
            
        start_fold = 26
        step = 4
        
        for fold_end in range(start_fold, n_weeks - step + 1, step):
            train_df = grp.iloc[:fold_end]
            test_df = grp.iloc[fold_end:fold_end+step]
            
            if test_df.empty: break
            
            # FIX P0: Patch test features to remove data leakage
            test_df = patch_test_features(test_df, train_df)
                
            bf = BayesianForecaster()
            if bf.fit(train_df):
                p10, p50, p90, _ = bf.predict(test_df)
            else:
                snf = SeasonalNaiveForecaster()
                snf.fit(train_df)
                p10, p50, p90, _ = snf.predict(test_df, train_df)
                
            # Segment-level collection
            all_actuals.extend(test_df['conversion_value'].values)
            all_p50s.extend(p50)
            all_p10s.extend(p10)
            all_p90s.extend(p90)
            
            # Channel roll-up collection
            fold_key = fold_end
            if fold_key not in channel_fold_data:
                channel_fold_data[fold_key] = {}
            if ch not in channel_fold_data[fold_key]:
                channel_fold_data[fold_key][ch] = {'actual': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
            
            channel_fold_data[fold_key][ch]['actual'] += test_df['conversion_value'].sum()
            channel_fold_data[fold_key][ch]['p50'] += np.sum(p50)
            channel_fold_data[fold_key][ch]['p10'] += np.sum(p10)
            channel_fold_data[fold_key][ch]['p90'] += np.sum(p90)
            
    all_actuals = np.array(all_actuals)
    all_p50s = np.array(all_p50s)
    all_p10s = np.array(all_p10s)
    all_p90s = np.array(all_p90s)
    
    if len(all_actuals) == 0:
        print("Not enough data for cross validation evaluation (need > 30 weeks per segment).")
        return
    
    # --- Segment-level (Channel × CampaignType) metrics ---
    mape = mean_absolute_percentage_error(all_actuals, all_p50s)
    rmse = np.sqrt(np.mean((all_actuals - all_p50s)**2))
    
    mask = all_actuals > 0
    mdape = np.median(np.abs((all_actuals[mask] - all_p50s[mask]) / all_actuals[mask])) * 100
    
    coverage = np.mean((all_actuals >= all_p10s) & (all_actuals <= all_p90s)) * 100
    
    # Weighted MAPE: weights errors by actual revenue
    wmape = np.sum(np.abs(all_actuals[mask] - all_p50s[mask])) / np.sum(all_actuals[mask]) * 100
    
    # --- Channel-level roll-up metrics (Stage 2) ---
    ch_actuals = []
    ch_p50s = []
    ch_p10s = []
    ch_p90s = []
    
    for fold_key, channels in channel_fold_data.items():
        for ch, vals in channels.items():
            ch_actuals.append(vals['actual'])
            ch_p50s.append(vals['p50'])
            ch_p10s.append(vals['p10'])
            ch_p90s.append(vals['p90'])
    
    ch_actuals = np.array(ch_actuals)
    ch_p50s = np.array(ch_p50s)
    ch_p10s = np.array(ch_p10s)
    ch_p90s = np.array(ch_p90s)
    
    ch_mask = ch_actuals > 0
    ch_mape = mean_absolute_percentage_error(ch_actuals, ch_p50s)
    ch_mdape = np.median(np.abs((ch_actuals[ch_mask] - ch_p50s[ch_mask]) / ch_actuals[ch_mask])) * 100
    ch_rmse = np.sqrt(np.mean((ch_actuals - ch_p50s)**2))
    ch_coverage = np.mean((ch_actuals >= ch_p10s) & (ch_actuals <= ch_p90s)) * 100
    
    # --- Account-level roll-up metrics ---
    acct_fold_data = {}
    for fold_key, channels in channel_fold_data.items():
        if fold_key not in acct_fold_data:
            acct_fold_data[fold_key] = {'actual': 0.0, 'p10': 0.0, 'p50': 0.0, 'p90': 0.0}
        for ch, vals in channels.items():
            acct_fold_data[fold_key]['actual'] += vals['actual']
            acct_fold_data[fold_key]['p50'] += vals['p50']
            acct_fold_data[fold_key]['p10'] += vals['p10']
            acct_fold_data[fold_key]['p90'] += vals['p90']
    
    acct_actuals = np.array([v['actual'] for v in acct_fold_data.values()])
    acct_p50s = np.array([v['p50'] for v in acct_fold_data.values()])
    acct_p10s = np.array([v['p10'] for v in acct_fold_data.values()])
    acct_p90s = np.array([v['p90'] for v in acct_fold_data.values()])
    
    acct_mask = acct_actuals > 0
    acct_mape = mean_absolute_percentage_error(acct_actuals, acct_p50s)
    acct_mdape = np.median(np.abs((acct_actuals[acct_mask] - acct_p50s[acct_mask]) / acct_actuals[acct_mask])) * 100 if np.any(acct_mask) else np.nan
    acct_coverage = np.mean((acct_actuals >= acct_p10s) & (acct_actuals <= acct_p90s)) * 100
    
    print("=" * 70)
    print("STAGE 1: Segment-Level (Channel × CampaignType) — Weekly Predictions")
    print("=" * 70)
    print(f"  Predictions:        {len(all_actuals)}")
    print(f"  MAPE:               {mape:.2f}%  (Target: < 15%)")
    print(f"  wMAPE:              {wmape:.2f}%  (Revenue-weighted)")
    print(f"  MdAPE:              {mdape:.2f}%")
    print(f"  RMSE:               {rmse:.2f}")
    print(f"  Coverage (P10-P90): {coverage:.2f}%  (Target: 75-85%)")
    print()
    print("=" * 70)
    print("STAGE 2: Channel-Level Roll-Up — 4-Week Block Predictions")
    print("=" * 70)
    print(f"  Predictions:        {len(ch_actuals)}")
    print(f"  MAPE:               {ch_mape:.2f}%")
    print(f"  MdAPE:              {ch_mdape:.2f}%")
    print(f"  RMSE:               {ch_rmse:.2f}")
    print(f"  Coverage (P10-P90): {ch_coverage:.2f}%")
    print()
    print("=" * 70)
    print("STAGE 2: Account-Level Roll-Up — 4-Week Block Predictions")
    print("=" * 70)
    print(f"  Predictions:        {len(acct_actuals)}")
    print(f"  MAPE:               {acct_mape:.2f}%")
    print(f"  MdAPE:              {acct_mdape:.2f}%")
    print(f"  Coverage (P10-P90): {acct_coverage:.2f}%")
    print()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_evaluation()
