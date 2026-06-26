import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Explorer | AIgnition", page_icon="🗄️", layout="wide")
st.title("Historical Data Explorer")

API_URL = st.session_state.get("API_URL", "http://localhost:8000/api")

@st.cache_data(ttl=300)
def fetch_historical():
    try:
        resp = requests.get(f"{API_URL}/historical")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return pd.DataFrame(data)
        return pd.DataFrame()
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
