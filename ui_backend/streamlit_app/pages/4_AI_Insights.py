import streamlit as st
import requests

st.set_page_config(page_title="AI Insights | AIgnition", page_icon="🧠", layout="wide")
st.title("Causal AI Insights")

API_URL = st.session_state.get("API_URL", "http://localhost:8000/api")

st.markdown("""
This module uses **LangGraph** to trigger a 9-node state machine that analyzes the Bayesian forecast outputs, detects anomalies, assesses confidence, and uses Groq (Qwen/Llama) to generate causal summaries and budget opportunities.
""")

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Generate Fresh Insights", type="primary"):
        with st.spinner("LangGraph is running... this may take a few seconds as it makes sequential LLM calls."):
            resp = requests.post(f"{API_URL}/insights")
            if resp.status_code == 200:
                st.session_state.insights = resp.json()
                st.success("Success!")
            else:
                st.error("Failed to generate insights.")

    if st.button("Load Last Insights"):
        resp = requests.get(f"{API_URL}/insights")
        if resp.status_code == 200:
            st.session_state.insights = resp.json()
        else:
            st.error("No insights found.")

with col2:
    if "insights" in st.session_state:
        data = st.session_state.insights
        
        conf = data.get('confidence_level', 'UNKNOWN')
        color = "green" if conf == "HIGH" else "orange" if conf == "MEDIUM" else "red"
        st.markdown(f"**Overall Forecast Confidence:** :{color}[**{conf}**]")
        
        st.info(f"**Executive Summary:**\n\n{data.get('summary', '')}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("**Key Drivers:**")
            for driver in data.get('key_drivers', []):
                st.markdown(f"- {driver}")
                
            st.success("**Budget Opportunities:**")
            for opp in data.get('opportunities', []):
                st.markdown(f"- {opp}")
                
        with c2:
            st.warning("**Identified Risks:**")
            for risk in data.get('risks', []):
                st.markdown(f"- {risk}")
                
            st.error("**Statistical Anomalies:**")
            for anom in data.get('anomalies', []):
                st.markdown(f"- {anom}")
                
        st.caption(f"Analysis generated at: {data.get('metadata', {}).get('analysis_timestamp')} using {data.get('metadata', {}).get('model_used')}")
    else:
        st.info("Click a button on the left to load or generate LangGraph insights.")
