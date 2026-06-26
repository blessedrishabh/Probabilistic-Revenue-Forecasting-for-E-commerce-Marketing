# Probabilistic Revenue & ROAS Forecasting Workflow
## AIgnition 3.0 — NetElixir Hackathon

---

## 0. Guiding Principles

| Principle | Decision |
|---|---|
| Forecast type | Probabilistic ranges (P10 / P50 / P90), never single point estimates |
| Forecast granularity | Aggregate period only — 30 / 60 / 90 days |
| Attribution source | Channel-level CSV data treated as ground truth; no re-attribution |
| Primary metrics | Revenue (aggregate + per channel) and Blended ROAS |
| MMM / custom attribution | Out of scope |
| Budget relationship | Revenue = f(spend, channel, seasonality, trend); modeled per channel |

---

## 1. Data Ingestion & Schema Normalization

### 1.1 Expected Input Files

```
/data/
  google_ads.csv
  meta_ads.csv
  microsoft_ads.csv
  ga4_sessions.csv       # source/medium level, optional enrichment
  shopify_orders.csv     # ground truth revenue
```

### 1.2 Canonical Schema (applied to all channel CSVs after normalization)

| Column | Type | Notes |
|---|---|---|
| `date` | DATE (YYYY-MM-DD) | Week-level aggregation preferred |
| `channel` | ENUM {google, meta, microsoft} | Normalized from raw source field |
| `campaign_type` | STR | e.g. Brand Search, Non-Brand Search, Shopping, PMAX, Retargeting |
| `campaign_name` | STR | Raw campaign identifier |
| `spend` | FLOAT | Total media cost in USD |
| `impressions` | INT | |
| `clicks` | INT | |
| `conversions` | FLOAT | Platform-reported conversions |
| `conversion_value` | FLOAT | Platform-reported revenue (channel attribution) |
| `roas_reported` | FLOAT | `conversion_value / spend`; computed at ingestion |

### 1.3 Normalization Steps

```
Step 1 — Date alignment
  - Detect date column format per file (MM/DD/YYYY, YYYY-MM-DD, etc.)
  - Parse and cast to ISO DATE
  - Aggregate to weekly bins (Monday as week start) to reduce daily noise
  - Fill missing weeks with zero spend / zero revenue (not NaN)

Step 2 — Channel tagging
  - Infer channel from filename or explicit source column
  - Map to canonical {google | meta | microsoft}

Step 3 — Campaign type inference
  - Apply rule-based regex classifier on campaign_name:
      Brand Search   → /brand|branded|[client_name]/i
      Non-Brand Search → /search|keyword/ AND NOT brand pattern
      Shopping / PLA → /shopping|pla|merchant/i
      PMAX           → /pmax|performance.max/i
      Retargeting    → /retarg|remarketing|rlsa/i
      Display        → /display|gdn|audience/i
      Video          → /video|youtube|reels|vma/i
  - Unmatched → tag as "Other"

Step 4 — Deduplication
  - Drop rows where spend = 0 AND conversion_value = 0 AND impressions = 0
  - Deduplicate on (date, channel, campaign_name) keeping last row

Step 5 — Revenue reconciliation
  - Compare SUM(google.conversion_value + meta.conversion_value + msft.conversion_value)
    vs SUM(shopify_orders.revenue) by week
  - Compute platform_shopify_ratio per channel per week
  - Flag weeks where ratio < 0.7 or > 1.5 as anomalous (used in AI layer, not corrected)
  - Use platform-reported conversion_value as the forecasting target (per brief: existing
    attribution is source of truth)
```

---

## 2. Exploratory Feature Construction

### 2.1 Channel-Level Derived Metrics (weekly grain)

```python
# Per (channel, campaign_type, week) row:

CPC       = spend / clicks               # cost per click
CVR       = conversions / clicks         # conversion rate
AOV       = conversion_value / conversions  # average order value
ROAS      = conversion_value / spend

# Efficiency ratios — used as model features
efficiency_index = ROAS / rolling_4w_ROAS  # current vs recent baseline

# Spend share within channel
spend_share = channel_spend / total_spend_that_week

# Log-transform spend and revenue (stabilizes variance for regression models)
log_spend   = log1p(spend)
log_revenue = log1p(conversion_value)
```

### 2.2 Seasonality Features

```
Week-of-year encoding
  - sin_week = sin(2π × week_of_year / 52)
  - cos_week = cos(2π × week_of_year / 52)

Month-of-year one-hot (M01 … M12)

Holiday / Peak period flags (binary)
  - us_holiday_week   → Thanksgiving (W47), Black Friday (W47), Cyber Monday (W48),
                        Christmas (W51/52), New Year (W01), Valentine (W07),
                        Memorial Day (W21), Labor Day (W36), Prime Day (W28-ish)
  - pre_holiday_week  → 1 week before each holiday_week
  - post_holiday_week → 1 week after each holiday_week

YoY growth multiplier (if ≥ 2 years of data available)
  - yoy_revenue_growth = revenue_this_week / revenue_same_week_last_year
  - Used as a direct scaling signal when data is available
```

### 2.3 Rolling / Lag Features (computed per channel–campaign_type pair)

```
lag_1w_revenue    = revenue shifted by 1 week
lag_2w_revenue    = revenue shifted by 2 weeks
lag_4w_revenue    = revenue shifted by 4 weeks
lag_52w_revenue   = revenue shifted by 52 weeks (YoY anchor)

rolling_4w_avg_revenue   = mean of last 4 weeks
rolling_8w_avg_revenue   = mean of last 8 weeks
rolling_4w_std_revenue   = std  of last 4 weeks  ← key for uncertainty estimation

rolling_4w_avg_ROAS      = mean ROAS over last 4 weeks
rolling_4w_avg_CVR       = mean CVR over last 4 weeks
rolling_4w_avg_CPC       = mean CPC over last 4 weeks
```

---

## 3. Forecasting Architecture

The system uses a **two-stage hierarchical approach**:

```
Stage 1 — Channel × CampaignType level forecasting
           (most granular forecast unit)

Stage 2 — Reconciliation & roll-up
           Channel Level   = SUM(campaign_types within channel)
           Account Level   = SUM(all channels)
           Blended ROAS    = Total Revenue / Total Spend
```

### 3.1 Model Selection Per Granularity

#### 3.1.1 Channel × CampaignType Level — Primary Forecasting Unit

**Model: Bayesian Ridge Regression with Monte Carlo sampling**

Why:
- Naturally outputs a posterior distribution → probabilistic intervals without bootstrapping overhead
- Works well on small datasets (typical hackathon data: 52–104 weekly rows per segment)
- Feature coefficients are interpretable for causal explanation layer
- Handles multicollinearity better than OLS (regularized)

```
Target variable:   log1p(conversion_value)   [reverse with expm1 at output]

Feature matrix X:
  [log1p(spend), sin_week, cos_week, lag_1w_revenue, lag_4w_revenue,
   rolling_4w_avg_ROAS, rolling_4w_avg_CVR, holiday_flag,
   pre_holiday_week, post_holiday_week, yoy_revenue_growth (if available)]

Training:
  - Fit one BayesianRidge model per (channel, campaign_type) segment
  - Use all available historical data; no train/val split for the final prototype
    (use time-series cross-validation for internal accuracy scoring only)

Prediction:
  - Generate 5000 posterior samples via BayesianRidge.predict() with return_std=True
  - Sample: y_samples = normal(mu_pred, sigma_pred, n_samples=5000)
  - Apply expm1() to convert back from log space
  - Extract P10 = 10th percentile, P50 = median, P90 = 90th percentile
```

#### 3.1.2 Fallback Model — Segments with < 8 Weeks of Data

**Model: Seasonal Naive with Uncertainty Bands**

```
P50 = rolling_4w_avg_revenue × seasonal_index × yoy_multiplier

seasonal_index = avg revenue in same month / overall avg revenue (computed from all available data)

Uncertainty bands:
  P10 = P50 × (1 - 1.5 × rolling_4w_cv)   where cv = std/mean
  P90 = P50 × (1 + 1.5 × rolling_4w_cv)
  Minimum band width: ±15% of P50
```

#### 3.1.3 Account-Level Cross-Check — Supplementary Model

**Model: Prophet (Meta's time-series library)**

```
Purpose: validate that bottom-up roll-up (Stage 2) is in the right ballpark

Input: weekly total revenue across all channels (single series)

Config:
  seasonality_mode = 'multiplicative'
  yearly_seasonality = True
  weekly_seasonality = False   (we're forecasting aggregate periods, not daily)
  holidays dataframe = US e-commerce holidays defined in 2.2

Output: account-level P10/P50/P90 for 30/60/90 day windows
        → used as sanity check only; bottom-up model is primary
```

### 3.2 ROAS Forecasting

ROAS is **not modeled directly** as a time series. It is derived from forecasted revenue and input spend:

```
For a given future_spend input per (channel, campaign_type):

  ROAS_P50  = Revenue_P50  / future_spend
  ROAS_P10  = Revenue_P10  / future_spend   (pessimistic ROAS)
  ROAS_P90  = Revenue_P90  / future_spend   (optimistic ROAS)

Blended ROAS (account level):
  Blended_ROAS_P50 = SUM(Revenue_P50 across all segments) / SUM(future_spend across all segments)
```

ROAS floor guardrail: flag any segment where ROAS_P50 < 1.0 as "spend at risk."

### 3.3 Forecast Windows

```
Forecast window requested: 30 / 60 / 90 days

Internal computation:
  - Convert days to integer weeks: n_weeks = ceil(days / 7)
    → 30d = 5 weeks, 60d = 9 weeks, 90d = 13 weeks

  - For each future week w:
      1. Update rolling features using predictions from prior future weeks
         (recursive multi-step forecasting)
      2. Apply seasonality features based on calendar position of week w
      3. Sample 5000 posterior draws per week
      4. Accumulate samples: period_revenue_samples = SUM(weekly_samples across n_weeks)

  - Aggregate period output:
      P10 = 10th percentile of period_revenue_samples
      P50 = 50th percentile
      P90 = 90th percentile
```

---

## 4. Budget Simulation Layer

This layer answers: *"What happens to Revenue and ROAS if I change spend?"*

### 4.1 Response Curve Estimation Per Channel × CampaignType

```
For each segment, fit a Diminishing Returns curve on historical (spend, revenue) pairs:

  Model: revenue = a × spend^b    (power law)
  Fit: log(revenue) = log(a) + b × log(spend)  via OLS on historical data
  Constraint: 0 < b < 1 (enforce diminishing returns; reject fits where b ≥ 1)

  Fallback (if <8 data points): use b = 0.75 (industry default for paid search)

  Marginal ROAS at spend S:
    dRevenue/dSpend = a × b × S^(b-1)

  This curve is used to score alternative budget allocations.
```

### 4.2 Budget Scenario Inputs

```
User provides per-scenario:
  - total_budget_30d (or 60d / 90d): total spend across all channels
  - Optional: channel_split = {google: 0.6, meta: 0.3, microsoft: 0.1}
    If not provided → use historical spend share as default

System generates three scenarios automatically:
  Scenario A (Conservative): 0.8 × current spend pace
  Scenario B (Baseline):     1.0 × current spend pace
  Scenario C (Aggressive):   1.2 × current spend pace

  Plus any user-custom scenario.
```

### 4.3 Scenario Forecast Generation

```
For each scenario:
  1. Distribute total_budget to channel × campaign_type using:
       a) user-specified channel_split, OR
       b) historical spend share
  2. For each segment, look up predicted revenue from response curve at the new spend level
  3. Apply seasonality multiplier for the forecast period
  4. Add uncertainty: propagate BayesianRidge posterior sigma scaled by new spend delta
  5. Aggregate to channel and account level
  6. Compute ROAS for each scenario

Output matrix:
  | Scenario | Channel | CampaignType | Spend | Rev_P10 | Rev_P50 | Rev_P90 | ROAS_P10 | ROAS_P50 | ROAS_P90 |
```

---

## 5. Campaign Consistency Validation

Run before forecasting. Block forecast generation on CRITICAL issues; warn on WARNING issues.

```
Check 1 — Date coverage gaps (CRITICAL if gap > 2 consecutive weeks with spend > 0)
  Flag: "Campaign {name} has {n}-week data gap from {start} to {end}. Forecasts for this
         segment will use fallback model."

Check 2 — Spend with zero conversions streaks (WARNING if ≥ 3 consecutive weeks)
  Flag: "Campaign {name} has {n} consecutive weeks of spend with zero attributed revenue.
         ROAS may be understated or attribution is broken."

Check 3 — Extreme ROAS outliers (WARNING if ROAS > 50 or ROAS < 0.3 in any single week)
  Flag: "Week {date}, Campaign {name}: ROAS = {val}. Possible data anomaly.
         Winsorizing at 3×IQR for model training."

Check 4 — Campaign name changes (WARNING if same campaign_type has >3 distinct names within a channel)
  Flag: "Channel {ch} has fragmented campaign naming in {type} type. Consider consolidating
         for more reliable segment-level forecasts."

Check 5 — Missing channel data (CRITICAL if any channel has zero weeks in the input)
  Flag: "No data found for {channel}. Revenue contribution from this channel will be
         excluded from aggregate forecast."

Check 6 — Spend ramp / structural break detection
  Compute Chow test on the revenue ~ spend regression at each historical midpoint.
  If break detected: use only data post-breakpoint for model training.
  Flag: "Structural change detected around {date} for {channel} {campaign_type}.
         Pre-break data excluded from training."
```

---

## 6. Accuracy Scoring (Internal Validation)

Run during development and report in technical documentation. Not shown to end users in prototype UI.

### 6.1 Time-Series Cross-Validation Protocol

```
Expanding window evaluation:

  Fold 1: Train on weeks 1–26, predict weeks 27–30  (30-day horizon)
  Fold 2: Train on weeks 1–30, predict weeks 31–34
  Fold 3: Train on weeks 1–34, predict weeks 35–38
  ... continue until 4 weeks remain

  For 60-day horizon: predict 9 weeks ahead
  For 90-day horizon: predict 13 weeks ahead
```

### 6.2 Metrics

```
Point accuracy (P50 evaluation):
  MAPE  = mean(|actual - P50| / actual) × 100        → target < 15%
  RMSE  = sqrt(mean((actual - P50)^2))
  MdAPE = median(|actual - P50| / actual) × 100      → robust to outliers

Probabilistic accuracy (interval evaluation):
  Coverage(P10, P90) = fraction of actuals falling within [P10, P90] → target: 75–85%
  Interval Width     = mean(P90 - P10) / P50         → measure of sharpness; lower is better
  Winkler Score      = width + (2/α) × penalty if actual outside interval  (α=0.2 for 80% interval)

ROAS-specific:
  ROAS_MAE  = mean(|actual_ROAS - predicted_ROAS_P50|)
  ROAS_MAPE = mean(|actual_ROAS - predicted_ROAS_P50| / actual_ROAS) × 100  → target < 20%

Channel-level accuracy tracked separately per {google, meta, microsoft}.
```

---

## 7. Output Schema

### 7.1 Aggregate Account-Level Forecast

```json
{
  "forecast_period_days": 30,
  "forecast_start_date": "2026-07-20",
  "forecast_end_date":   "2026-08-18",
  "scenario": "Baseline",
  "total_budget_input": 150000.00,
  "revenue": {
    "P10": 420000,
    "P50": 510000,
    "P90": 605000,
    "currency": "USD"
  },
  "blended_roas": {
    "P10": 2.80,
    "P50": 3.40,
    "P90": 4.03
  },
  "confidence_note": "80% probability interval. Actual outcome expected within P10–P90 range."
}
```

### 7.2 Channel-Level Forecast

```json
{
  "channel": "google",
  "budget_allocated": 90000,
  "revenue": { "P10": 250000, "P50": 305000, "P90": 365000 },
  "roas":    { "P10": 2.78,   "P50": 3.39,   "P90": 4.06  },
  "campaign_types": [
    {
      "campaign_type": "Brand Search",
      "budget_allocated": 20000,
      "revenue": { "P10": 72000, "P50": 88000, "P90": 105000 },
      "roas":    { "P10": 3.60,  "P50": 4.40,  "P90": 5.25  }
    },
    {
      "campaign_type": "Shopping",
      "budget_allocated": 40000,
      "revenue": { "P10": 108000, "P50": 130000, "P90": 156000 },
      "roas":    { "P10": 2.70,   "P50": 3.25,   "P90": 3.90  }
    }
  ]
}
```

### 7.3 Campaign-Level Forecast (Drill-Down)

```json
{
  "channel": "google",
  "campaign_type": "Shopping",
  "campaign_name": "GS | Shopping | All Products",
  "budget_allocated": 18000,
  "revenue": { "P10": 46000, "P50": 56000, "P90": 68000 },
  "roas":    { "P10": 2.56,  "P50": 3.11,  "P90": 3.78  },
  "flags": ["ROAS_BELOW_2_IN_PRIOR_PERIOD"]
}
```

---

## 8. AI-Assisted Causal Inference Layer

This layer runs **after** statistical forecasts are generated. It uses an LLM API (Anthropic Claude) to interpret patterns and communicate forecast drivers to non-technical users.

### 8.1 Structured Prompt Construction

For each forecast output, construct a prompt containing:

```
SYSTEM:
  You are an expert digital marketing analyst specializing in ecommerce performance
  forecasting. Analyze the structured data below and return a business-readable
  forecast explanation. Focus on causal reasoning, not generic statements.
  Return JSON with keys: summary, key_drivers, risks, opportunities, anomalies.

USER:
  ## Historical Performance Summary (last 12 weeks)
  [Insert: channel × campaign_type weekly revenue, spend, ROAS as a table]

  ## Detected Anomalies (from validation layer)
  [Insert: validation flag messages from Section 5]

  ## Structural Break Events
  [Insert: any detected breakpoints and their estimated revenue impact]

  ## Forecast Output
  [Insert: JSON from Section 7 for the requested period]

  ## Seasonal Context
  [Insert: list of holidays / peak events falling within the forecast window]

  ## Budget Scenario Applied
  [Insert: scenario name, total budget, channel allocation]

  Task:
  1. Explain in 3–5 sentences why revenue is expected to reach the P50 range.
  2. Identify the top 2 channel or campaign-type drivers of this forecast.
  3. List 2 operational risks that could push outcome toward P10.
  4. List 1–2 budget optimization opportunities based on response curve analysis.
  5. Flag any anomalies in historical data that reduce forecast confidence.
```

### 8.2 LLM Output Schema

```json
{
  "summary": "Revenue forecast of $510K (±$90K) for the next 30 days is driven primarily by ...",
  "key_drivers": [
    "Google Shopping campaigns have shown consistent 3.2× ROAS over the past 8 weeks, ...",
    "Seasonal uplift expected in Week 3 of the forecast window due to Prime Day proximity ..."
  ],
  "risks": [
    "Meta Retargeting ROAS has declined 22% over the past 4 weeks, suggesting audience ...",
    "A 3-week data gap in Microsoft Ads (Jun 1–21) creates uncertainty in that channel's ..."
  ],
  "opportunities": [
    "Google Brand Search marginal ROAS at current spend level is estimated at 6.8×, ...",
    "Reallocating 10% of Meta spend to Google Shopping could increase P50 revenue by ~$18K ..."
  ],
  "anomalies": [
    "Week of May 12: Google Shopping revenue spike (ROAS = 7.2×) appears inconsistent with ..."
  ]
}
```

### 8.3 When to Trigger LLM Calls

```
Trigger 1: Every new forecast generation (always)
  → Full causal summary for the selected period + scenario

Trigger 2: Anomaly detected in validation layer (always)
  → Targeted anomaly explanation prompt

Trigger 3: User changes budget scenario (always)
  → Incremental prompt comparing new scenario vs baseline
    (Include delta in P50 revenue and ROAS, ask LLM to explain the marginal change)

Trigger 4: Forecast confidence is low (P90 - P10 > 40% of P50)
  → Risk amplification prompt: ask LLM to identify specific weeks / segments
    contributing to wide uncertainty bands
```

---

## 9. Uncertainty Quantification Summary

| Source of Uncertainty | How Modeled |
|---|---|
| Historical revenue variance | BayesianRidge posterior sigma |
| Seasonality mismatch (no 52w anchor) | Wider prior on seasonal coefficients; flagged in output |
| Data gaps / anomalies | Fallback model with wider ±IQR bands; flagged in output |
| Multi-step forecast error accumulation | Monte Carlo simulation with step-to-step error carry-forward |
| Channel attribution inconsistency | Noted as qualitative risk in LLM causal summary |
| Budget response curve uncertainty | Power-law CI propagated through scenario revenue estimates |

Minimum reported interval width: P90 − P10 ≥ 20% of P50, regardless of model sigma.
This floor prevents false precision in low-variance historical periods.

---

## 10. Pipeline Execution Order

```
[1] Ingest raw CSVs
      ↓
[2] Schema normalization & weekly aggregation
      ↓
[3] Campaign consistency validation
      → CRITICAL errors: halt, return validation report
      → WARNINGs: continue, store flags for AI layer
      ↓
[4] Feature engineering (Section 2)
      ↓
[5] Structural break detection (Chow test per segment)
      → Trim pre-break data if break confirmed
      ↓
[6] Model fitting per (channel × campaign_type) segment
      → BayesianRidge if n_weeks ≥ 8
      → Seasonal Naive fallback if n_weeks < 8
      ↓
[7] Response curve fitting per segment (Section 4.1)
      ↓
[8] Receive forecast inputs:
      - forecast_window: 30 | 60 | 90
      - budget_scenario: {total_budget, optional channel_split}
      ↓
[9] Multi-step recursive forecasting with Monte Carlo sampling
      ↓
[10] Aggregate: campaign → campaign_type → channel → account
      ↓
[11] Compute ROAS at all levels
      ↓
[12] Cross-check account-level against Prophet output
       → If delta > 20%: log warning; surface in LLM risk flags
      ↓
[13] Construct LLM prompt with all context (Section 8.1)
      ↓
[14] Call Anthropic API; parse causal summary (Section 8.2)
      ↓
[15] Assemble final output (Section 7) + causal summary
      ↓
[16] Return structured JSON to frontend / reporting layer
```

---

## 11. Model Limitations & Assumptions

```
ASSUMPTION 1: Platform-reported conversion_value is the forecasting target.
  Revenue is never reconciled to Shopify ground truth in the model itself;
  Shopify data is used for anomaly flagging only.

ASSUMPTION 2: Spend → Revenue relationship is stable within each segment.
  Structural break detection (Step 5) mitigates but does not eliminate this risk.

ASSUMPTION 3: YoY seasonality pattern is representative.
  If only 1 year of data exists, seasonal patterns for Q4 will be extrapolated
  from Q1–Q3 data using sin/cos encoding. Confidence in Q4 forecasts will be lower.

ASSUMPTION 4: Channel mix (spend allocation) follows user input or historical share.
  Inter-channel cannibalization and incrementality effects are not modeled.

ASSUMPTION 5: No external demand shocks (competitor actions, platform policy changes,
  macroeconomic events) are anticipated. LLM causal layer can flag known risks if
  user provides context, but the statistical model cannot anticipate unknowns.

LIMITATION 1: Campaign-level forecasts with < 8 weeks of history use fallback model
  with significantly wider uncertainty intervals. Treat P10/P90 as rough bounds only.

LIMITATION 2: Microsoft Ads tends to have lower data volume; its contribution to
  blended ROAS will have the widest intervals of the three channels.

LIMITATION 3: The response curve (Section 4.1) is estimated from observational data.
  It will underestimate saturation at spend levels well above historical maximum.
```

---

## 12. Technology Stack (Forecasting Layer)

```
Language:        Python 3.11+

Core libraries:
  pandas          → data ingestion, normalization, feature engineering
  numpy           → numerical operations, Monte Carlo sampling
  scikit-learn    → BayesianRidge, preprocessing, cross-validation
  scipy.stats     → Chow test, interval computations
  prophet         → account-level sanity check model
  statsmodels     → power-law response curve OLS

LLM Integration:
  anthropic       → claude-sonnet-4-6 via /v1/messages

Forecast output:
  JSON (flat + nested)
  Optional: CSV export of scenario comparison matrix
```

---

*Document version: 1.0 | AIgnition 3.0 Hackathon | June 2026*
