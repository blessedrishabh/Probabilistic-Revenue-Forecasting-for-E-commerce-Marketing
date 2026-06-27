import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Budget Simulator | AIgnition", page_icon="🎛️", layout="wide")
st.title("Budget Simulator")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

@st.cache_data(ttl=60)
def fetch_base():
    fpath = os.path.join(BASE_DIR, 'Forecasting', 'forecast_output.json')
    if not os.path.exists(fpath):
        return None
    with open(fpath, 'r') as f:
        data = json.load(f)
    return {
        "forecast_period_days": data.get("forecast_period_days", 30),
        "scenario": data.get("scenario", "Baseline"),
        "total_budget": data.get("total_budget_input", 0.0),
        "revenue": data.get("revenue", {}),
        "blended_roas": data.get("blended_roas", {}),
        "channels": data.get("channels", {})
    }

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
        
        # if "last_sim_payload" not in st.session_state or st.session_state.last_sim_payload != payload:
        #     with st.spinner("Simulating..."):
        #         resp = requests.post(f"{API_URL}/simulate", json=payload)
        #         if resp.status_code == 200:
        #             st.session_state.sim_result = resp.json()
        #             st.session_state.last_sim_payload = payload
        #         else:
        #             st.error(f"Simulation failed: {resp.text}")
        if "last_sim_payload" not in st.session_state or st.session_state.last_sim_payload != payload:
            # Run simulation locally
            base_budget = base_data['total_budget']
            elasticity = 0.75
            channels_data = base_data.get('channels', {})

            # Calculate splits
            splits = {}
            for ch, ch_d in channels_data.items():
                splits[ch] = ch_d.get("budget_allocated", 0) / base_budget if base_budget > 0 else 0

            sim_channels = {}
            total_rev_p10, total_rev_p50, total_rev_p90 = 0, 0, 0

            for ch, ch_d in channels_data.items():
                base_ch_budget = ch_d.get("budget_allocated", 0)
                new_ch_budget = new_budget * splits.get(ch, 0)
                multiplier = (new_ch_budget / base_ch_budget) ** elasticity if base_ch_budget > 0 else 0

                np10 = ch_d["revenue"]["P10"] * multiplier
                np50 = ch_d["revenue"]["P50"] * multiplier
                np90 = ch_d["revenue"]["P90"] * multiplier

                sim_channels[ch] = {
                    "budget_allocated": round(new_ch_budget, 2),
                    "revenue": {"P10": round(np10, 2), "P50": round(np50, 2), "P90": round(np90, 2)},
                    "roas": {
                        "P10": round(np10 / new_ch_budget, 2) if new_ch_budget > 0 else 0,
                        "P50": round(np50 / new_ch_budget, 2) if new_ch_budget > 0 else 0,
                        "P90": round(np90 / new_ch_budget, 2) if new_ch_budget > 0 else 0,
                    }
                }
                total_rev_p10 += np10
                total_rev_p50 += np50
                total_rev_p90 += np90

            st.session_state.sim_result = {
                "scenario": scenario,
                "total_budget": round(new_budget, 2),
                "total_revenue": {"P10": round(total_rev_p10, 2), "P50": round(total_rev_p50, 2), "P90": round(total_rev_p90, 2)},
                "blended_roas": {
                    "P10": round(total_rev_p10 / new_budget, 2) if new_budget > 0 else 0,
                    "P50": round(total_rev_p50 / new_budget, 2) if new_budget > 0 else 0,
                    "P90": round(total_rev_p90 / new_budget, 2) if new_budget > 0 else 0,
                },
                "channels": sim_channels
            }
            st.session_state.last_sim_payload = payload

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
