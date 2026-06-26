import os
import re
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def infer_campaign_type(campaign_name):
    if not isinstance(campaign_name, str):
        return "Other"
    cn = campaign_name.lower()
    if re.search(r'brand|branded|netelixir', cn):
        return "Brand Search"
    elif re.search(r'search|keyword', cn):
        return "Non-Brand Search"
    elif re.search(r'shopping|pla|merchant', cn):
        return "Shopping"
    elif re.search(r'pmax|performance\.max', cn):
        return "PMAX"
    elif re.search(r'retarg|remarketing|rlsa', cn):
        return "Retargeting"
    elif re.search(r'display|gdn|audience', cn):
        return "Display"
    elif re.search(r'video|youtube|reels|vma', cn):
        return "Video"
    else:
        return "Other"

def deduplicate_and_select(df):
    if df.empty: return df
    df = df.sort_values('date')
    df = df.drop_duplicates(subset=['date', 'channel', 'campaign_name'], keep='last')
    mask = (df['spend'] == 0) & (df['conversion_value'] == 0) & (df['impressions'] == 0)
    df = df[~mask]
    cols = ['date', 'channel', 'campaign_type', 'campaign_name', 'spend', 'impressions', 'clicks', 'conversions', 'conversion_value']
    return df[cols]

def load_google_ads(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['segments_date'])
    df['channel'] = 'google'
    df['campaign_type'] = df['campaign_name'].apply(infer_campaign_type)
    df['spend'] = df['metrics_cost_micros'] / 1e6
    df['impressions'] = df['metrics_impressions']
    df['clicks'] = df['metrics_clicks']
    df['conversions'] = df['metrics_conversions']
    df['conversion_value'] = df['metrics_conversions_value']
    return deduplicate_and_select(df)

def load_meta_ads(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date_start'])
    df['channel'] = 'meta'
    df['campaign_type'] = df['campaign_name'].apply(infer_campaign_type)
    df['spend'] = df['spend']
    df['impressions'] = df['impressions']
    df['clicks'] = df['clicks']
    # FIX P2: Derive conversions from conversion_value (Meta doesn't provide a count)
    df['conversions'] = (df['conversion'] > 0).astype(float)
    df['conversion_value'] = df['conversion'] 
    return deduplicate_and_select(df)

def load_microsoft_ads(filepath):
    if not os.path.exists(filepath): return pd.DataFrame()
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['TimePeriod'])
    df['channel'] = 'microsoft'
    df['campaign_type'] = df['CampaignName'].apply(infer_campaign_type)
    df['campaign_name'] = df['CampaignName']
    df['spend'] = df['Spend']
    df['impressions'] = df['Impressions']
    df['clicks'] = df['Clicks']
    df['conversions'] = df['Conversions']
    df['conversion_value'] = df['Revenue']
    return deduplicate_and_select(df)

# FIX 1: Restore Channel × CampaignType granularity
def pad_missing_weeks(weekly):
    if weekly.empty: return weekly
    
    padded_list = []
    for (ch, ct), grp in weekly.groupby(['channel', 'campaign_type']):
        active_grp = grp[grp['spend'] > 0]
        if active_grp.empty:
            continue
        min_date = active_grp['week_start'].min()
        max_date = active_grp['week_start'].max()
        all_weeks = pd.date_range(min_date, max_date, freq='W-MON')
        
        grp = grp.set_index('week_start')
        grp = grp.reindex(all_weeks)
        grp['channel'] = ch
        grp['campaign_type'] = ct
        grp.fillna({
            'spend': 0, 'impressions': 0, 'clicks': 0, 'conversions': 0, 'conversion_value': 0
        }, inplace=True)
        padded_list.append(grp.reset_index(names='week_start'))
        
    return pd.concat(padded_list, ignore_index=True)

def aggregate_to_weekly(df):
    if df.empty: return df
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='d')
    agg_funcs = {
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'conversion_value': 'sum'
    }
    # FIX 1: Aggregate at channel × campaign_type (not campaign_name, not channel-only)
    weekly = df.groupby(['week_start', 'channel', 'campaign_type'], as_index=False).agg(agg_funcs)
    weekly = pad_missing_weeks(weekly)
    weekly['roas_reported'] = np.where(weekly['spend'] > 0, weekly['conversion_value'] / weekly['spend'], 0)
    return weekly
