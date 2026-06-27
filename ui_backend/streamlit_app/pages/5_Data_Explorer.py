import streamlit as st
import json
import os
import sys
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Explorer | AIgnition", page_icon="🗄️", layout="wide")
st.title("Historical Data Explorer")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

@st.cache_data(ttl=300)
def fetch_historical():
    try:
        from Forecasting.pipeline import load_google_ads, load_meta_ads, load_microsoft_ads, aggregate_to_weekly
        df_g = load_google_ads(os.path.join(DATA_DIR, 'google_ads_campaign_stats.csv'))
        df_m = load_meta_ads(os.path.join(DATA_DIR, 'meta_ads_campaign_stats.csv'))
        df_b = load_microsoft_ads(os.path.join(DATA_DIR, 'bing_campaign_stats.csv'))
        df_all = pd.concat([df_g, df_m, df_b], ignore_index=True)
        weekly_df = aggregate_to_weekly(df_all)
        weekly_df['week_start'] = weekly_df['week_start'].astype(str)
        return weekly_df.sort_values(['week_start', 'channel'])
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return pd.DataFrame()

df = fetch_historical()

if not df.empty and 'channel' in df.columns:
    channels = df['channel'].unique().tolist()
    selected_channels = st.multiselect("Filter by Channel", channels, default=channels)
    
    filtered_df = df[df['channel'].isin(selected_channels)]
    
    # Chart
    st.subheader("Historical Revenue & Spend Trend")
    
    # Aggregate by week
    trend = filtered_df.groupby('week_start')[['spend', 'conversion_value']].sum().reset_index()
    trend = trend.rename(columns={'conversion_value': 'Revenue', 'spend': 'Spend'})
    
    fig = px.line(trend, x='week_start', y=['Revenue', 'Spend'], 
                  labels={'value': 'Amount ($)', 'variable': 'Metric', 'week_start': 'Week'},
                  template="plotly_dark",
                  title="Weekly Revenue & Spend")
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Channel-level breakdown
    st.subheader("Revenue by Channel Over Time")
    ch_trend = filtered_df.groupby(['week_start', 'channel'])['conversion_value'].sum().reset_index()
    fig2 = px.line(ch_trend, x='week_start', y='conversion_value', color='channel',
                   labels={'conversion_value': 'Revenue ($)', 'week_start': 'Week'},
                   template="plotly_dark")
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Raw Data Table")
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.error("No historical data available. Ensure FastAPI backend is running and data files exist.")
