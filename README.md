# 🚦 AI-Driven Parking & Congestion Intelligence

An AI-driven system for detecting illegal-parking hotspots in Bengaluru and quantifying their impact on traffic flow, built to enable **targeted, prioritized enforcement** instead of reactive, experience-driven patrolling.

**Live demo:** _[add your Streamlit Cloud URL here after deploying]_

---

## The Problem

On-street illegal parking near commercial areas, metro stations, and junctions chokes carriageways — but today:
- Enforcement is patrol-based and reactive, not data-driven.
- There is no heatmap linking parking violations to actual congestion impact.
- There is no way to prioritize *where* limited enforcement resources should go.

Working with ~6 months of real, anonymized Bengaluru traffic-violation records, two findings shaped this entire project:
- **0%** of violations in the dataset have any recorded closure or action-taken timestamp — there is currently no visibility into whether flagged violations are ever resolved.
- **~28.7%** of auto-flagged violations are rejected on human validation — meaning roughly 1 in 3 flags are false positives, a real cost to any naive "respond to every flag" enforcement strategy.

## Approach

1. **Data pipeline** (`src/pipeline.py`) — cleans raw violation records, keeps only human-validated (`approved`) records, and assigns a **domain-informed severity score** per violation type (e.g. double-parking and road-crossing obstructions score higher than footpath parking, since they create a more direct physical chokepoint). Locations are indexed using **Uber H3** spatial hexagons for consistent, scale-appropriate spatial aggregation.

2. **Feature engineering** (`src/features.py`) — aggregates violation-level data into a spatio-temporal master table: one row per `(H3 cell, hour, day-of-week)`, with total violations, aggregated severity, and mean coordinates.

3. **Forecasting model** (`src/train.py`) — trains both a solo LightGBM regressor and a stacked LightGBM+XGBoost ensemble to predict congestion severity from temporal/spatial features, **automatically selecting whichever model earns its complexity** (the ensemble is only shipped if it beats solo LightGBM by a meaningful margin; current run shipped solo LightGBM after the ensemble didn't clear the bar).

4. **Dashboard** (`app.py`) — a Streamlit app with:
   - A 3D hexagon map (PyDeck/H3) of historical or forecasted severity, log-scaled for visibility across a long-tailed severity distribution.
   - A **predictive engine** that forecasts severity for any selected hour/day combination, not just historically observed ones.
   - A **What-If Patrol Simulator** — model the effect of deploying N patrol interceptors on citywide severity (illustrative suppression assumption, stated explicitly, not a measured causal effect).
   - **SHAP-based explainability** — shows which features (hour, day, historical violation volume) drove a zone's severity score.
   - An **LLM-generated deployment briefing** (Gemini) for the top-priority zone.
   - A **rule-based statutory reference card** mapping severity tier to the relevant Motor Vehicles Act section — a static lookup, explicitly *not* retrieval-augmented generation.

## Model Performance

See `models/training_metrics.json` for the exact numbers from the most recent training run (RMSE / MAE / R² for both the solo and ensemble models, and which one was shipped).

## Known Limitations & Honest Assumptions

- This is a **historical-pattern system**, not a live feed — there is no real-time camera/sensor integration. Forecasts extrapolate from historical hour/day patterns, not live traffic conditions.
- Severity weights per violation type are **domain judgment**, not derived from measured ground-truth congestion data (no live traffic-speed dataset was available).
- The patrol simulator's "6% reduction per interceptor" is an **illustrative assumption** for demonstrating the what-if mechanic, not a measured causal effect.
- Motor Vehicles Act section citations in the Statutory Reference Card are **indicative** and should be verified against current state notifications before any real operational use.

## Future Work

- Integrate a live violation feed instead of batch historical data.
- Correlate severity scores against real measured congestion/travel-time data (e.g. via a traffic API) to validate the scoring methodology empirically.
- Explore sovereign/local map data sources (e.g. Mappls, Bhuvan) for India-specific deployments.
- Predict `validation_status` (approved vs. rejected) to reduce false-positive patrol dispatches.

## Setup

```bash
git clone <your-repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (never commit this file):
```
GEMINI_API_KEY=your_key_here
```

Run the data pipeline once to regenerate processed data and the trained model (optional — only needed if you don't already have `data/master_table.csv` and `models/ensemble_forecaster.pkl`):
```bash
python src/pipeline.py
python src/features.py
python src/train.py
```

Launch the dashboard:
```bash
streamlit run app.py
```

## Tech Stack

Python · Streamlit · PyDeck (deck.gl) · LightGBM · XGBoost · scikit-learn · SHAP · Uber H3 · Google Gemini API

## Data Source

Anonymized traffic-violation records, Bengaluru, provided for this hackathon. Raw row-level data is not included in this repository — only the aggregated, non-identifying spatio-temporal master table.