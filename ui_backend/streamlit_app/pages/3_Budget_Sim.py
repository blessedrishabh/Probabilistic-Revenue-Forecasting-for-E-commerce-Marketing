import streamlit as st
import requests

st.set_page_config(page_title="Budget Simulator | AIgnition", page_icon="🎛️", layout="wide")
st.title("Budget Simulator")

API_URL = st.session_state.get("API_URL", "http://localhost:8000/api")

@st.cache_data(ttl=60)
def fetch_base():
    resp = requests.get(f"{API_URL}/forecast")
    if resp.status_code == 200:
        return resp.json()
    return None

base_data = fetch_base()

if base_data:
    st.markdown("Adjust the total budget and run the simulation to see how revenue changes based on historical spend-to-revenue elasticity.")
    
    current_budget = base_data['total_budget']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        
        # Initialize widget states if they don't exist
        if 'sim_scenario_widget' not in st.session_state:
            st.session_state.sim_scenario_widget = "Custom"
        if 'sim_slider_widget' not in st.session_state:
            st.session_state.sim_slider_widget = int(current_budget)

        def on_scenario_change():
            scen = st.session_state.sim_scenario_widget
            if scen == "Conservative":
                st.session_state.sim_slider_widget = int(current_budget * 0.8)
            elif scen == "Baseline":
                st.session_state.sim_slider_widget = int(current_budget * 1.0)
            elif scen == "Aggressive":
                st.session_state.sim_slider_widget = int(current_budget * 1.2)

        def on_slider_change():
            st.session_state.sim_scenario_widget = "Custom"

        scenario = st.radio(
            "Scenario Name", 
            ["Conservative", "Baseline", "Aggressive", "Custom"], 
            key="sim_scenario_widget",
            on_change=on_scenario_change
        )
        
        new_budget = st.slider(
            "Total Budget ($)", 
            min_value=int(current_budget * 0.5), 
            max_value=int(current_budget * 2.0), 
            format="$%d",
            key="sim_slider_widget",
            on_change=on_slider_change
        )
        
        budget_delta = new_budget - current_budget
        pct_change = (budget_delta / current_budget * 100) if current_budget > 0 else 0
        st.caption(f"Change: **${budget_delta:+,.0f}** ({pct_change:+.1f}%)")
        
        # Run simulation automatically
        payload = {
            "scenario": scenario,
            "total_budget": float(new_budget)
        }
        
        if "last_sim_payload" not in st.session_state or st.session_state.last_sim_payload != payload:
            with st.spinner("Simulating..."):
                resp = requests.post(f"{API_URL}/simulate", json=payload)
                if resp.status_code == 200:
                    st.session_state.sim_result = resp.json()
                    st.session_state.last_sim_payload = payload
                else:
                    st.error(f"Simulation failed: {resp.text}")

    with col2:
        st.subheader("Results")
        
        if "sim_result" in st.session_state:
            sim = st.session_state.sim_result
            
            # Comparison metrics
            st.markdown("#### Current vs Simulated")
            sc1, sc2, sc3 = st.columns(3)
            
            delta_rev = sim['total_revenue']['P50'] - base_data['revenue']['P50']
            delta_roas = sim['blended_roas']['P50'] - base_data['blended_roas']['P50']
            
            sc1.metric(
                "Budget", 
                f"${sim['total_budget']:,.0f}", 
                delta=f"${(sim['total_budget'] - current_budget):,.0f}",
                delta_color="off"
            )
            sc2.metric(
                "P50 Revenue", 
                f"${sim['total_revenue']['P50']:,.0f}", 
                delta=f"${delta_rev:,.0f}"
            )
            sc3.metric(
                "P50 ROAS", 
                f"{sim['blended_roas']['P50']:.2f}x", 
                delta=f"{delta_roas:+.2f}x"
            )
            
            st.markdown("#### Channel-Level Simulation")
            ch_data = []
            for ch, d in sim['channels'].items():
                roas_flag = "⚠️ Spend at Risk" if d['roas']['P50'] < 1.0 else ""
                ch_data.append({
                    "Channel": ch.capitalize(),
                    "Budget ($)": f"${d['budget_allocated']:,.2f}",
                    "Rev P10": f"${d['revenue']['P10']:,.2f}",
                    "Rev P50": f"${d['revenue']['P50']:,.2f}",
                    "Rev P90": f"${d['revenue']['P90']:,.2f}",
                    "ROAS": f"{d['roas']['P50']:.2f}x",
                    "Flag": roas_flag
                })
            st.dataframe(ch_data, use_container_width=True)
        else:
            st.info("Adjust the budget slider or scenario to see projected outcomes.")
else:
    st.error("Could not load baseline forecast. Ensure FastAPI is running.")
