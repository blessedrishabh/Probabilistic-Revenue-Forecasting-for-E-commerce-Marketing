import os
import json
import logging
from datetime import timedelta
import pandas as pd
import numpy as np

from pipeline import load_google_ads, load_meta_ads, load_microsoft_ads, aggregate_to_weekly
from feature_engineering import compute_features
from models import BayesianForecaster, SeasonalNaiveForecaster
from budget_simulator import ResponseCurve, generate_scenarios

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_pipeline(forecast_period_days=30):
    logging.info(f"Running pipeline for {forecast_period_days} days forecast period.")
    logging.info("Ingesting raw CSVs...")
    df_google = load_google_ads('data/google_ads_campaign_stats.csv')
    df_meta = load_meta_ads('data/meta_ads_campaign_stats.csv')
    df_ms = load_microsoft_ads('data/bing_campaign_stats.csv')
    
    df_all = pd.concat([df_google, df_meta, df_ms], ignore_index=True)
    if df_all.empty:
        logging.error("No data loaded. Check file paths.")
        return
        
    logging.info("Schema normalization & weekly aggregation...")
    weekly_df = aggregate_to_weekly(df_all)
    
    logging.info("Feature engineering...")
    features_df = compute_features(weekly_df)
    
    # FIX 1: Stage 1 — Forecast at Channel × CampaignType level
    logging.info("Model fitting and response curve estimation...")
    segments = features_df.groupby(['channel', 'campaign_type'])
    models = {}
    curves = {}
    
    for (ch, ct), grp in segments:
        bf = BayesianForecaster()
        if not bf.fit(grp):
            snf = SeasonalNaiveForecaster()
            snf.fit(grp)
            models[(ch, ct)] = ('snf', snf, grp)
        else:
            models[(ch, ct)] = ('bayesian', bf, grp)
            
        rc = ResponseCurve()
        rc.fit(grp)
        curves[(ch, ct)] = rc

    # FIX 5: Recursive multi-step forecasting
    logging.info(f"Forecast generation (Baseline scenario, {forecast_period_days} days)...")
    
    forecast_results = {
        "forecast_period_days": forecast_period_days,
        "scenario": "Baseline",
        "channels": {}
    }
    
    # Collect all segment-level Monte Carlo samples for hierarchical roll-up
    channel_samples = {}   # channel -> accumulated revenue samples
    channel_budgets = {}
    total_revenue_samples = np.zeros((5000,))
    total_budget_input = 0.0
    segment_details = {}   # (ch, ct) -> details dict
    
    for (ch, ct), (mod_type, model, grp) in models.items():
        # simple baseline: project last 4 weeks avg spend
        recent_avg_weekly_spend = grp['spend'].tail(4).mean()
        
        # Scale the budget for the requested period
        period_weeks = forecast_period_days / 7.0
        forecast_spend = recent_avg_weekly_spend * period_weeks
        if pd.isna(forecast_spend): forecast_spend = 0.0
        
        rc = curves[(ch, ct)]
        
        # FIX 5: Recursive multi-step — predict one week at a time, feed predictions back as lags
        n_future_weeks = int(round(period_weeks))
        if n_future_weeks < 1: n_future_weeks = 1
        future_dates = pd.date_range(grp['week_start'].max() + timedelta(days=7), periods=n_future_weeks, freq='W-MON')
        
        all_week_samples = []
        
        # Initialize running lag values from the tail of historical data
        running_lag_1w = grp['conversion_value'].iloc[-1] if len(grp) > 0 else 0
        running_lag_4w = grp['conversion_value'].iloc[-4] if len(grp) >= 4 else grp['conversion_value'].mean()
        running_rolling_4w = grp['conversion_value'].tail(4).values.tolist()
        
        for w_idx, future_date in enumerate(future_dates):
            future_row = pd.DataFrame({'week_start': [future_date]})
            
            if mod_type == 'bayesian':
                future_row['log_spend'] = np.log1p(forecast_spend / period_weeks)
                # Use last known clicks/impressions as proxy
                future_row['log_clicks'] = grp['log_clicks'].iloc[-1] if 'log_clicks' in grp.columns else 0
                future_row['log_impressions'] = grp['log_impressions'].iloc[-1] if 'log_impressions' in grp.columns else 0
                woy = future_date.isocalendar()[1]
                future_row['sin_week'] = np.sin(2 * np.pi * woy / 52)
                future_row['cos_week'] = np.cos(2 * np.pi * woy / 52)
                
                # FIX 5: Use recursively updated lags
                future_row['lag_1w_revenue'] = running_lag_1w
                future_row['lag_4w_revenue'] = running_lag_4w
                
                # lag_52w: use historical value from 52 weeks ago if available
                target_52w_date = future_date - timedelta(weeks=52)
                hist_52w = grp[grp['week_start'] == target_52w_date]
                future_row['lag_52w_revenue'] = hist_52w['conversion_value'].values[0] if not hist_52w.empty else 0
                
                # Rolling averages from last available 
                future_row['rolling_4w_avg_ROAS'] = grp['rolling_4w_avg_ROAS'].iloc[-1] if not grp.empty else 0
                future_row['rolling_4w_avg_CVR'] = grp['rolling_4w_avg_CVR'].iloc[-1] if not grp.empty else 0
                future_row['rolling_4w_avg_CPC'] = grp['rolling_4w_avg_CPC'].iloc[-1] if 'rolling_4w_avg_CPC' in grp.columns and not grp.empty else 0
                future_row['rolling_8w_avg_revenue'] = grp['rolling_8w_avg_revenue'].iloc[-1] if 'rolling_8w_avg_revenue' in grp.columns and not grp.empty else 0
                
                # Holiday flags
                holiday_weeks = {47, 48, 51, 52, 1, 7, 21, 36, 28}
                future_row['holiday_flag'] = 1 if woy in holiday_weeks else 0
                future_row['pre_holiday_week'] = 1 if (woy + 1) in holiday_weeks else 0
                future_row['post_holiday_week'] = 1 if (woy - 1) in holiday_weeks else 0
                
                # YoY growth: use last available
                future_row['yoy_revenue_growth'] = grp['yoy_revenue_growth'].iloc[-1] if 'yoy_revenue_growth' in grp.columns and not grp.empty else 1.0
                
                # New features added
                future_row['lag_2w_revenue'] = running_rolling_4w[-2] if len(running_rolling_4w) >= 2 else 0
                future_row['rolling_4w_std_revenue'] = float(np.std(running_rolling_4w)) if len(running_rolling_4w) > 1 else 0.0
                future_row['spend_share'] = grp['spend_share'].iloc[-1] if 'spend_share' in grp.columns and not grp.empty else 0
                future_row['efficiency_index'] = grp['efficiency_index'].iloc[-1] if 'efficiency_index' in grp.columns and not grp.empty else 0
                future_row['spend_momentum'] = grp['spend_momentum'].iloc[-1] if 'spend_momentum' in grp.columns and not grp.empty else 1.0
                future_row['lag1_x_sin'] = future_row['lag_1w_revenue'] * future_row['sin_week']
                future_row['lag1_x_cos'] = future_row['lag_1w_revenue'] * future_row['cos_week']
                for m in range(1, 13):
                    future_row[f'M{m:02d}'] = 1 if future_date.month == m else 0
                
                p10, p50, p90, week_samples = model.predict(future_row, n_samples=5000)
                week_rev_samples = week_samples[0]  # shape: (5000,)
            else:
                p10, p50, p90, week_samples = model.predict(future_row, grp)
                week_rev_samples = week_samples[0]
            
            all_week_samples.append(week_rev_samples)
            
            # FIX 5: Update running lags for next iteration
            predicted_revenue = float(np.median(week_rev_samples))
            running_lag_4w = running_lag_1w  # what was 1w ago becomes 4w ago in 3 more steps (approximate)
            running_lag_1w = predicted_revenue
            running_rolling_4w.append(predicted_revenue)
            if len(running_rolling_4w) > 4:
                running_rolling_4w.pop(0)
        
        # Sum weekly samples to get 30-day period samples
        segment_rev_samples = np.sum(all_week_samples, axis=0)  # shape: (5000,)
        
        p10_val = max(0, np.percentile(segment_rev_samples, 10))
        p50_val = max(0, np.percentile(segment_rev_samples, 50))
        p90_val = max(0, np.percentile(segment_rev_samples, 90))
        
        # FIX 4: Enforce minimum interval width at segment level too
        min_width = 0.20 * p50_val
        if (p90_val - p10_val) < min_width and p50_val > 0:
            half_deficit = (min_width - (p90_val - p10_val)) / 2
            p90_val += half_deficit
            p10_val = max(0, p10_val - half_deficit)
        
        # Store segment-level detail
        segment_details[(ch, ct)] = {
            "campaign_type": ct,
            "budget_allocated": round(forecast_spend, 2),
            "revenue": {
                "P10": round(p10_val, 2),
                "P50": round(p50_val, 2),
                "P90": round(p90_val, 2)
            },
            "roas": {
                "P10": round(p10_val / forecast_spend, 2) if forecast_spend > 0 else 0,
                "P50": round(p50_val / forecast_spend, 2) if forecast_spend > 0 else 0,
                "P90": round(p90_val / forecast_spend, 2) if forecast_spend > 0 else 0
            }
        }
        
        # FIX 1: Stage 2 — Accumulate samples for channel-level roll-up
        if ch not in channel_samples:
            channel_samples[ch] = np.zeros((5000,))
            channel_budgets[ch] = 0.0
        
        channel_samples[ch] += segment_rev_samples
        channel_budgets[ch] += forecast_spend
        total_budget_input += forecast_spend
        total_revenue_samples += segment_rev_samples

    # FIX 1: Stage 2 — Compute channel-level P10/P50/P90 from rolled-up samples
    for ch in channel_samples:
        ch_rev = channel_samples[ch]
        ch_bud = channel_budgets[ch]
        
        ch_p10 = max(0, np.percentile(ch_rev, 10))
        ch_p50 = max(0, np.percentile(ch_rev, 50))
        ch_p90 = max(0, np.percentile(ch_rev, 90))
        
        # Collect campaign_type details for this channel
        ch_campaign_types = []
        for (seg_ch, seg_ct), detail in segment_details.items():
            if seg_ch == ch:
                ch_campaign_types.append(detail)
        
        forecast_results['channels'][ch] = {
            "budget_allocated": round(ch_bud, 2),
            "revenue": {
                "P10": round(ch_p10, 2),
                "P50": round(ch_p50, 2),
                "P90": round(ch_p90, 2)
            },
            "roas": {
                "P10": round(ch_p10 / ch_bud, 2) if ch_bud > 0 else 0,
                "P50": round(ch_p50 / ch_bud, 2) if ch_bud > 0 else 0,
                "P90": round(ch_p90 / ch_bud, 2) if ch_bud > 0 else 0
            },
            "campaign_types": ch_campaign_types
        }

    # Account-level totals from rolled-up samples
    forecast_results['total_budget_input'] = round(total_budget_input, 2)
    forecast_results['revenue'] = {
        "P10": round(max(0, np.percentile(total_revenue_samples, 10)), 2),
        "P50": round(max(0, np.percentile(total_revenue_samples, 50)), 2),
        "P90": round(max(0, np.percentile(total_revenue_samples, 90)), 2),
        "currency": "USD"
    }
    forecast_results['blended_roas'] = {
        "P10": round(forecast_results['revenue']['P10'] / total_budget_input, 2) if total_budget_input > 0 else 0,
        "P50": round(forecast_results['revenue']['P50'] / total_budget_input, 2) if total_budget_input > 0 else 0,
        "P90": round(forecast_results['revenue']['P90'] / total_budget_input, 2) if total_budget_input > 0 else 0
    }

    with open('forecast_output.json', 'w') as f:
        json.dump(forecast_results, f, indent=2)

    logging.info("Forecasting pipeline complete. Output saved to forecast_output.json.")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run forecasting pipeline.")
    parser.add_argument('--days', type=int, default=30, choices=[30, 60, 90], 
                        help='Forecasting window in days (30, 60, or 90)')
    args = parser.parse_args()
    
    run_pipeline(forecast_period_days=args.days)
