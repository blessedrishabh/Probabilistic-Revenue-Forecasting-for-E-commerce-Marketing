import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard | AIgnition", page_icon="📊", layout="wide")

API_URL = st.session_state.get("API_URL", "http://localhost:8000/api")

colA, colB = st.columns([3, 1])
with colA:
    st.title("Account Overview")
with colB:
    st.markdown("<br>", unsafe_allow_html=True)
    f_days = st.selectbox("Forecast Window", [30, 60, 90], index=0, label_visibility="collapsed")
    if st.button("Update Forecast", use_container_width=True):
        with st.spinner(f"Generating {f_days}-day forecast..."):
            try:
                requests.post(f"{API_URL}/forecast/generate?days={f_days}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error("Failed to generate forecast")

@st.cache_data(ttl=60)
def fetch_overview():
    try:
        response = requests.get(f"{API_URL}/forecast")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_channels():
    try:
        response = requests.get(f"{API_URL}/forecast/channels")
        response.raise_for_status()
        return response.json().get("channels", {})
    except Exception as e:
        return {}

@st.cache_data(ttl=60)
def fetch_insights():
    try:
        response = requests.get(f"{API_URL}/insights")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

data = fetch_overview()
channels = fetch_channels()
insights = fetch_insights()

if data:
    st.markdown(f"### {data['scenario']} Scenario | {data['forecast_period_days']}-Day Forecast")
    
    # Confidence badge from AI insights
    if insights:
        conf = insights.get("confidence_level", "UNKNOWN")
        color = "green" if conf == "HIGH" else "orange" if conf == "MEDIUM" else "red"
        st.markdown(f"**Forecast Confidence:** :{color}[**{conf}**]")
    
    st.markdown("---")
    
    # High Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Budget", f"${data['total_budget']:,.0f}")
    with col2:
        st.metric("P50 Revenue (Expected)", f"${data['revenue']['P50']:,.0f}")
    with col3:
        st.metric("P10 Revenue (Conservative)", f"${data['revenue']['P10']:,.0f}")
    with col4:
        st.metric("P90 Revenue (Upside)", f"${data['revenue']['P90']:,.0f}")

    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.metric("Blended ROAS (P50)", f"{data['blended_roas']['P50']:.2f}x")
    with rcol2:
        st.metric("Blended ROAS (P10)", f"{data['blended_roas']['P10']:.2f}x")
    with rcol3:
        st.metric("Blended ROAS (P90)", f"{data['blended_roas']['P90']:.2f}x")

    st.markdown("---")
    
    # Channel Breakdown Chart
    st.subheader("Revenue Contribution by Channel")
    if channels:
        ch_names = [ch.capitalize() for ch in channels.keys()]
        p10s = [c["revenue"]["P10"] for c in channels.values()]
        p50s = [c["revenue"]["P50"] for c in channels.values()]
        p90s = [c["revenue"]["P90"] for c in channels.values()]
        budgets = [c["budget_allocated"] for c in channels.values()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ch_names, y=p10s, name='P10 (Conservative)', marker_color='#94A3B8'))
        fig.add_trace(go.Bar(x=ch_names, y=p50s, name='P50 (Expected)', marker_color='#4F46E5'))
        fig.add_trace(go.Bar(x=ch_names, y=p90s, name='P90 (Upside)', marker_color='#10B981'))
        
        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="Revenue ($)",
            xaxis_title="Channel"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel ROAS comparison
    if channels:
        st.subheader("ROAS by Channel")
        ch_names = [ch.capitalize() for ch in channels.keys()]
        roas_p50 = [c["roas"]["P50"] for c in channels.values()]
        
        fig2 = go.Figure()
        colors = ['#EF4444' if r < 1.0 else '#10B981' if r > 3.0 else '#F59E0B' for r in roas_p50]
        fig2.add_trace(go.Bar(x=ch_names, y=roas_p50, marker_color=colors, text=[f"{r:.2f}x" for r in roas_p50], textposition='outside'))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Break-even (1.0x)")
        fig2.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title="ROAS",
            xaxis_title="Channel",
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AI Summary snippet
    if insights:
        st.markdown("---")
        st.subheader("AI Executive Summary")
        st.info(insights.get("summary", "No summary available."))
        st.caption("Navigate to **AI Insights** page for full analysis.")
else:
    st.error("Could not load forecast data. Ensure FastAPI backend is running on port 8000.")
