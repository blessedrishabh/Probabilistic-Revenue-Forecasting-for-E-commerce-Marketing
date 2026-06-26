# AIgnition Forecasting & Causal Inference

AIgnition is an advanced predictive analytics and decision-support platform designed to optimize omni-channel marketing spend. It leverages Bayesian time-series forecasting, Monte Carlo simulations, and LLM-powered causal insights to bridge the gap between raw marketing data and executive decision-making.

---

## ✨ Key Features

- **Hierarchical Bayesian Forecasting:** Predicts future revenue across channels (Google, Meta, Microsoft) and campaign types using historical spend, seasonality, ad-stock effects, and rolling efficiency metrics.
- **Dynamic Budget Simulation:** Interactively manipulate your total budget allocations using an interactive Streamlit dashboard. Evaluate standard scenarios (Conservative, Baseline, Aggressive) or use a custom budget slider to instantly view projected P10, P50, and P90 revenue outcomes.
- **AI-Powered Executive Insights:** A LangGraph integration analyzes numerical anomalies, budget changes, and ROAS elasticity, outputting a clear, human-readable executive summary directly to the dashboard.
- **Interactive Streamlit Dashboard:** A sleek, responsive UI combining Plotly data visualizations with real-time forecasting.
- **Headless FastAPI Backend:** Fully decoupled API architecture serving forecast overviews, historical data, and on-demand simulations.

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit, Plotly
- **Backend API:** FastAPI, Uvicorn
- **Machine Learning & Stats:** Pandas, NumPy, Statsmodels, Scikit-learn
- **AI / LLM Layer:** LangChain, LangGraph, Groq (Llama-3.1-8b-instant)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AIgnition-Forecasting.git
cd AIgnition-Forecasting
```

### 2. Set Up a Virtual Environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Generate the Baseline Forecast
Before launching the UI, generate the baseline `forecast_output.json` file. You can specify a forecast window of 30, 60, or 90 days:
```bash
cd Forecasting
python main.py --days 30
cd ..
```

---

## 🏃 Running the Application

This architecture requires two separate servers running concurrently: the headless FastAPI backend and the Streamlit frontend UI.

### Start the FastAPI Backend
Open a terminal and run:
```bash
python ui_backend/run.py
```
*The API will be available at `http://localhost:8000`*

### Start the Streamlit Dashboard
Open a **second** terminal window and run:
```bash
python -m streamlit run ui_backend/streamlit_app/app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`*

---

## 📂 Project Structure

```text
├── Forecasting/                 # Machine learning & simulation models
│   ├── data/                    # Raw marketing CSV datasets
│   ├── budget_simulator.py      # Response curve elasticity math
│   ├── feature_engineering.py   # Transformation of raw ad data
│   ├── main.py                  # Core pipeline script
│   └── models.py                # Bayesian and Seasonal Naive models
├── llm_Integration/             # LangGraph LLM processing layer
│   ├── causal_analyzer.py       # Core LangGraph state machine graph
│   └── config.py                # LLM provider configuration
├── ui_backend/                  # API and Frontend code
│   ├── api/                     # FastAPI route definitions
│   ├── streamlit_app/           # Streamlit dashboard pages
│   └── run.py                   # Uvicorn entry point
└── README.md
```
