import numpy as np
import pandas as pd

def compute_features(weekly_df):
    if weekly_df.empty: return weekly_df
    
    weekly_df = weekly_df.sort_values(['channel', 'campaign_type', 'week_start'])
    
    weekly_df['CPC'] = np.where(weekly_df['clicks'] > 0, weekly_df['spend'] / weekly_df['clicks'], 0)
    weekly_df['CVR'] = np.where(weekly_df['clicks'] > 0, weekly_df['conversions'] / weekly_df['clicks'], 0)
    weekly_df['AOV'] = np.where(weekly_df['conversions'] > 0, weekly_df['conversion_value'] / weekly_df['conversions'], 0)
    weekly_df['ROAS'] = weekly_df['roas_reported']
    
    weekly_total_spend = weekly_df.groupby('week_start')['spend'].transform('sum')
    weekly_channel_spend = weekly_df.groupby(['week_start', 'channel'])['spend'].transform('sum')
    weekly_df['spend_share'] = np.where(weekly_total_spend > 0, weekly_channel_spend / weekly_total_spend, 0)
    
    weekly_df['log_spend'] = np.log1p(weekly_df['spend'])
    weekly_df['log_revenue'] = np.log1p(weekly_df['conversion_value'])
    
    # FIX P3: Add log_clicks and log_impressions as spend-proxy features
    weekly_df['log_clicks'] = np.log1p(weekly_df['clicks'])
    weekly_df['log_impressions'] = np.log1p(weekly_df['impressions'])
    
    weekly_df['week_of_year'] = weekly_df['week_start'].dt.isocalendar().week.astype(int)
    weekly_df['sin_week'] = np.sin(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['cos_week'] = np.cos(2 * np.pi * weekly_df['week_of_year'] / 52)
    
    # Month one-hot encoding
    weekly_df['month'] = weekly_df['week_start'].dt.month
    for m in range(1, 13):
        weekly_df[f'M{m:02d}'] = (weekly_df['month'] == m).astype(int)
    
    holiday_weeks = {47, 48, 51, 52, 1, 7, 21, 36, 28}
    weekly_df['holiday_flag'] = weekly_df['week_of_year'].isin(holiday_weeks).astype(int)
    weekly_df['pre_holiday_week'] = weekly_df['week_of_year'].apply(lambda w: 1 if (w + 1) in holiday_weeks else 0)
    weekly_df['post_holiday_week'] = weekly_df['week_of_year'].apply(lambda w: 1 if (w - 1) in holiday_weeks else 0)
    
    # Group at channel × campaign_type
    grouped = weekly_df.groupby(['channel', 'campaign_type'])
    
    # --- Lag features (already correctly shifted — no leakage) ---
    weekly_df['lag_1w_revenue'] = grouped['conversion_value'].shift(1).fillna(0)
    weekly_df['lag_2w_revenue'] = grouped['conversion_value'].shift(2).fillna(0)
    weekly_df['lag_4w_revenue'] = grouped['conversion_value'].shift(4).fillna(0)
    weekly_df['lag_52w_revenue'] = grouped['conversion_value'].shift(52).fillna(0)
    
    # --- FIX P1: Rolling features — shift(1) to exclude current row ---
    # Before: rolling included current row (leakage). After: only uses past data.
    weekly_df['rolling_4w_avg_revenue'] = grouped['conversion_value'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()).shift(1).fillna(0)
    weekly_df['rolling_4w_std_revenue'] = grouped['conversion_value'].transform(
        lambda x: x.rolling(4, min_periods=1).std()).shift(1).fillna(0)
    weekly_df['rolling_8w_avg_revenue'] = grouped['conversion_value'].transform(
        lambda x: x.rolling(8, min_periods=1).mean()).shift(1).fillna(0)
    weekly_df['rolling_4w_avg_ROAS'] = grouped['ROAS'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()).shift(1).fillna(0)
    weekly_df['rolling_4w_avg_CVR'] = grouped['CVR'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()).shift(1).fillna(0)
    weekly_df['rolling_4w_avg_CPC'] = grouped['CPC'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()).shift(1).fillna(0)
    
    # --- FIX P0: efficiency_index — use LAGGED ROAS (not current) ---
    lagged_ROAS = grouped['ROAS'].shift(1).fillna(0)
    weekly_df['efficiency_index'] = np.where(
        weekly_df['rolling_4w_avg_ROAS'] > 0, 
        lagged_ROAS / weekly_df['rolling_4w_avg_ROAS'], 0)

    # --- FIX P0: YoY growth — use lagged values only (no current revenue) ---
    # "What was the YoY growth rate LAST week?" — purely causal
    lag_1w = grouped['conversion_value'].shift(1).fillna(0)
    lag_53w = grouped['conversion_value'].shift(53).fillna(0)
    weekly_df['yoy_revenue_growth'] = np.where(
        lag_53w > 0,
        lag_1w / lag_53w,
        1.0
    )
    weekly_df['yoy_revenue_growth'] = weekly_df['yoy_revenue_growth'].clip(0.1, 10.0)

    # Spend momentum: use shifted spend rolling average
    rolling_4w_spend = grouped['spend'].transform(
        lambda x: x.rolling(4, min_periods=1).mean()).shift(1).fillna(0)
    weekly_df['spend_momentum'] = np.where(
        rolling_4w_spend > 0,
        weekly_df['spend'] / (rolling_4w_spend + 1e-5),
        1.0
    )
    weekly_df['spend_momentum'] = weekly_df['spend_momentum'].clip(0.1, 10.0)

    # Lag interaction: lag_1w * seasonal signal
    weekly_df['lag1_x_sin'] = weekly_df['lag_1w_revenue'] * weekly_df['sin_week']
    weekly_df['lag1_x_cos'] = weekly_df['lag_1w_revenue'] * weekly_df['cos_week']
                                              
    return weekly_df
