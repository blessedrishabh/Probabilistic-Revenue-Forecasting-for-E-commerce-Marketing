import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Channel Drilldown | AIgnition", page_icon="🔍", layout="wide")
st.title("Channel Drilldown")

API_URL = st.session_state.get("API_URL", "http://localhost:8000/api")

@st.cache_data(ttl=60)
def fetch_channels():
    try:
        response = requests.get(f"{API_URL}/forecast/channels")
        response.raise_for_status()
        return response.json().get("channels", {})
    except Exception as e:
        st.error("Failed to load channels")
        return {}

channels = fetch_channels()

if channels:
    selected_ch = st.selectbox("Select Marketing Channel", options=list(channels.keys()))
    ch_data = channels[selected_ch]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Budget Allocated", f"${ch_data['budget_allocated']:,.2f}")
    col2.metric("P50 Revenue", f"${ch_data['revenue']['P50']:,.2f}")
    col3.metric("P50 ROAS", f"{ch_data['roas']['P50']:.2f}x")
    
    st.markdown("### Campaign Types")
    campaigns = ch_data.get("campaign_types", [])
    
    if campaigns:
        df = pd.json_normalize(campaigns)
        # Clean up column names for display
        df = df.rename(columns={
            "campaign_type": "Campaign Type",
            "budget_allocated": "Budget ($)",
            "revenue.P10": "Rev P10",
            "revenue.P50": "Rev P50",
            "revenue.P90": "Rev P90",
            "roas.P50": "ROAS"
        })
        
        # Select specific columns
        display_cols = ["Campaign Type", "Budget ($)", "Rev P10", "Rev P50", "Rev P90", "ROAS"]
        df_display = df[display_cols]
        
        st.dataframe(
            df_display.style.format({
                "Budget ($)": "${:,.2f}",
                "Rev P10": "${:,.2f}",
                "Rev P50": "${:,.2f}",
                "Rev P90": "${:,.2f}",
                "ROAS": "{:.2f}x"
            }),
            use_container_width=True
        )
        
        # Bubble chart: Budget vs ROAS, size = Revenue
        st.markdown("### Efficiency Matrix")
        fig = px.scatter(
            df_display, x="Budget ($)", y="ROAS", size="Rev P50", color="Campaign Type",
            hover_name="Campaign Type", size_max=60,
            template="plotly_dark",
            title=f"{selected_ch.capitalize()} Efficiency"
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
