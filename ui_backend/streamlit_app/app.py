import streamlit as st
import os

# Must be the first Streamlit command
st.set_page_config(
    page_title="AIgnition Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3233/3233036.png", width=50)
st.sidebar.title("AIgnition")
st.sidebar.caption("Forecasting & Causal Inference v1.0")

st.markdown("""
# Welcome to AIgnition 📈

This dashboard connects directly to the Bayesian forecasting engine and LangGraph AI integration layer.

**Please select a page from the sidebar to begin.**
- **Dashboard**: High-level account metrics and blended ROAS
- **Channels**: Drill down into specific marketing segments
- **Budget Sim**: Interactively adjust budget and see predicted outcomes
- **AI Insights**: Generate and view LangGraph-powered causal summaries
- **Data Explorer**: View the raw historical dataset
""")

# API Configuration
if "API_URL" not in st.session_state:
    st.session_state.API_URL = "http://localhost:8000/api"
