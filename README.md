---
title: Probabilistic Revenue Forecasting
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

# 📈 Probabilistic Revenue Forecasting for E-Commerce Marketing

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B.svg?style=flat&logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E.svg?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-AI_Insights-1C3C3C.svg?style=flat&logo=langchain)](https://langchain-ai.github.io/langgraph/)

A production-grade, end-to-end Bayesian forecasting system that ingests omni-channel marketing data (Google Ads, Meta Ads, Microsoft Ads), generates **probabilistic revenue predictions with calibrated uncertainty intervals (P10/P50/P90)**, and delivers LLM-powered causal summaries — all through an interactive Streamlit dashboard.

---

## 🎯 The Problem

Marketing teams pour budgets across Google, Meta, and Microsoft campaigns, but have no reliable way to answer: *"If I spend X next month, what revenue range should I actually expect?"* Point-estimate forecasts hide uncertainty. This system replaces guesswork with mathematically honest confidence intervals so executives know the **best case, expected case, and worst case** before committing a single dollar.

---

## 🏗️ Architecture & Engineering Principles

This project was built following **Bayesian-first** and **Mathematical Honesty** principles.

1. **Probabilistic, Not Point-Estimate:** Every forecast is a full posterior distribution via `BayesianRidge`, not a single number. We draw 5,000 Monte Carlo samples per segment and report P10/P50/P90 revenue intervals so stakeholders see the true uncertainty envelope.
2. **Hierarchical Roll-Up:** Forecasts are generated at the **Channel × Campaign Type** level (e.g., Google → PMAX, Meta → Retargeting), then statistically aggregated upward to channel-level and account-level totals — preserving correlation structure across segments.
3. **Recursive Multi-Step Forecasting:** Instead of a single jump prediction, the system forecasts one week at a time, feeding predicted values back as lag features for the next week. This prevents the classic "flat-line" failure of naïve multi-step models.
4. **Zero-Inflation Handling:** Sparse segments (e.g., Microsoft Shopping) with >50% zero-revenue weeks get a dampened zero-injection layer on the Monte Carlo samples, preventing the model from hallucinating revenue where none historically existed.
5. **AI-Powered Causal Summaries:** A 9-node LangGraph state machine analyzes the numerical forecast outputs, detects anomalies, assesses confidence, and uses Groq (Qwen-2.5-32b) to generate executive-ready causal narratives and budget reallocation opportunities.

---

## ✨ Key Features

- **Bayesian Revenue Forecasting:** Predicts future revenue across 3 channels and 12+ campaign types using 20+ engineered features including ad-stock lags, seasonality harmonics, holiday flags, YoY growth, and rolling efficiency metrics.
- **Dynamic Budget Simulator:** Adjust total budget with a slider and instantly see projected P10/P50/P90 revenue outcomes using fitted OLS response curves with diminishing-returns elasticity (default α = 0.75).
- **AI Executive Insights:** LangGraph orchestrates sequential LLM calls to produce a structured JSON report: executive summary, key revenue drivers, identified risks, statistical anomalies, and concrete budget reallocation opportunities.
- **Interactive Streamlit Dashboard:** 5-page clinical-grade UI combining Plotly visualizations with real-time forecasting — Account Overview, Channel Drilldown, Budget Simulator, AI Insights, and Historical Data Explorer.
- **Hackathon-Ready Pipeline:** Single-command execution via `run.sh` with pre-trained pickle artifacts, pinned dependencies, and reproducible seeds.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.12 |
| **ML & Stats** | Scikit-Learn (BayesianRidge), Statsmodels (OLS), NumPy, Pandas, SciPy |
| **AI / LLM** | LangChain, LangGraph, Groq API (Qwen-2.5-32b-it) |
| **Frontend** | Streamlit, Plotly |
| **Deployment** | Hugging Face Spaces, GitHub |

---

## 🚀 How to Run the Project (Step-by-Step)

### ⚡ Quick Start (For Evaluators)
```bash
git clone https://github.com/blessedrishabh/Probabilistic-Revenue-Forecasting-for-E-commerce-Marketing.git
cd Probabilistic-Revenue-Forecasting-for-E-commerce-Marketing
pip install -r requirements.txt
bash run.sh
```
Output will be at `output/predictions.csv`. No API keys or extra setup needed.

> **Note:** Steps 2, 3, and 5 below are for **local development only** (AI Insights dashboard). The core forecasting pipeline runs fully offline with no internet or API keys.

### Full Local Setup
To run this entire system end-to-end on your local machine, follow these instructions.

### Step 1: Environment Setup
Clone the repository and install the locked dependencies to guarantee reproducibility.
```bash
git clone https://github.com/blessedrishabh/Probabilistic-Revenue-Forecasting-for-E-commerce-Marketing.git
cd Probabilistic-Revenue-Forecasting-for-E-commerce-Marketing

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux

# Install all dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API keys:
```env
GROQ_API_KEY_1=your_groq_api_key_here
GROQ_API_KEY_2=your_second_groq_api_key_here
```
*Note: The AI Insights feature requires valid Groq API keys. All other features work without them.*

### Step 3: Train the Bayesian Pipeline (One-Time Setup)
This step ingests the raw CSV data, applies schema normalization, engineers 20+ features, fits the BayesianRidge models per segment, and serializes everything to a pickle artifact.
```bash
python Forecasting/train.py
```
*You should see "Models successfully saved to pickle/model.pkl" in the terminal.*

### Step 4: Generate the Baseline Forecast
Load the pre-trained model and produce probabilistic predictions. You can specify a forecast window of 30, 60, or 90 days:
```bash
python Forecasting/main.py --data_dir ./data --model_path ./pickle/model.pkl --output_path ./output/predictions.csv
```
Or simply run the hackathon entry point:
```bash
bash run.sh
```

### Step 5: Launch the Dashboard
```bash
streamlit run ui_backend/streamlit_app/app.py
```
*A browser window will automatically open at `http://localhost:8501`. You can now view account-level metrics, drill into channels, simulate budget scenarios, and read AI-generated causal insights!*

---

## 📂 Project Structure

```text
├── run.sh                       # Single-command entry point (hackathon pipeline)
├── requirements.txt             # Pinned Python dependencies
├── app.py                       # Hugging Face Spaces entry point
├── data/                        # Raw marketing CSV datasets (overwritten at test time)
│   ├── google_ads_campaign_stats.csv
│   ├── meta_ads_campaign_stats.csv
│   └── bing_campaign_stats.csv
├── pickle/                      # Pre-trained model artifacts
│   └── model.pkl                # Serialized BayesianRidge models + ResponseCurves
├── Forecasting/                 # Core ML pipeline
│   ├── train.py                 # Phase 1: Train & serialize models
│   ├── main.py                  # Phase 2: Load model → predict → export CSV
│   ├── pipeline.py              # Schema normalization & weekly aggregation
│   ├── feature_engineering.py   # 20+ feature transformations (lags, seasonality, etc.)
│   ├── models.py                # BayesianForecaster & SeasonalNaiveForecaster classes
│   └── budget_simulator.py      # OLS response curve with diminishing-returns elasticity
├── llm_Integration/             # LangGraph AI processing layer
│   ├── graph.py                 # 9-node state machine for causal analysis
│   ├── config.py                # Groq API & model configuration
│   ├── nodes/                   # Individual LangGraph node implementations
│   └── prompts/                 # LLM prompt templates
├── ui_backend/                  # Frontend application
│   └── streamlit_app/           # Streamlit dashboard
│       ├── app.py               # Main app entry with sidebar navigation
│       ├── pages/               # 5 dashboard pages
│       │   ├── 1_Dashboard.py   # Account-level metrics & channel breakdown
│       │   ├── 2_Channels.py    # Campaign-type drilldown with efficiency matrix
│       │   ├── 3_Budget_Sim.py  # Interactive budget simulator
│       │   ├── 4_AI_Insights.py # LangGraph causal summaries
│       │   └── 5_Data_Explorer.py # Historical trend visualization
│       └── styles/              # Custom CSS
└── README.md
```

---

## 🔬 Model Details & Methodology

| Component | Implementation |
|-----------|---------------|
| **Primary Model** | `BayesianRidge` with `RobustScaler` — produces posterior mean + sigma for each prediction |
| **Fallback Model** | `SeasonalNaiveForecaster` — activated when a segment has <8 data points |
| **Uncertainty** | 5,000 Monte Carlo samples drawn from N(μ, σ) in log-space, then expm1-transformed |
| **Bias Correction** | Conditional log-normal correction: +0.5σ² only when σ < 1.0 to prevent over-prediction in volatile segments |
| **Zero Inflation** | Segments with >50% zero weeks get dampened zero-injection (50% of historical zero rate) |
| **Response Curves** | OLS log-log regression: Revenue = a × Spend^b, with elasticity capped at (0, 1) to enforce diminishing returns |
| **Interval Floor** | Adaptive minimum interval width scaled by model uncertainty: `clip(σ × 0.8, 0.15, 0.50) × P50` |

---

## ⚠️ Deployment Notes

- **Hugging Face Spaces:** The dashboard runs as a standalone Streamlit app reading directly from pre-generated JSON artifacts. No backend server required.
- **Hackathon Pipeline:** The automated testing system runs `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv` — this loads the pickle, generates features from the test data, and writes predictions to CSV.
- **LLM Dependency:** The AI Insights page requires valid Groq API keys. All forecasting and simulation features work fully offline with no network calls.
- **Reproducibility:** Random seed is set to `42` via `np.random.seed(42)` before all prediction runs. All dependency versions are pinned in `requirements.txt`.
