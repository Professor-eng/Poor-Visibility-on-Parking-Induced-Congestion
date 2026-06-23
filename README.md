# AI-Driven Parking & Congestion Intelligence

An AI-driven system for detecting illegal parking hotspots in Bengaluru and measuring their impact on traffic congestion.

**Live demo:** https://poor-visibility-on-parking-induced-congestion-rydynvc8bnzzvcmb.streamlit.app/

---

## The Problem

On-street illegal parking and spillover parking near commercial areas, metro stations, and events choke carriageways and intersections.
**Why It’s Hard Today**
- Enforcement is patrol-based and reactive.
- No heatmap of parking violations vs. congestion impact.
- Difficult to prioritize enforcement zones.

## Approach

1. **Data pipeline** (`src/pipeline.py`) — cleans raw violation records, keeps only approved records, and assigns a **severity score** per violation type (e.g. double-parking and road crossing obstructions score higher than footpath parking, since they create a more direct physical chokepoint). Locations are indexed using **Uber H3** spatial hexagons.

2. **Feature engineering** (`src/features.py`) — aggregates violation data into one row per `(H3 cell, hour, day-of-week)`, with total violations, aggregated severity, and mean coordinates.

3. **Forecasting model** (`src/train.py`) — trains both a solo LightGBM regressor and a stacked LightGBM+XGBoost ensemble to predict congestion severity from, **automatically selecting whichever model earns its complexity** (the ensemble is only shipped if it beats solo LightGBM by a meaningful margin; current run shipped solo LightGBM after the ensemble didn't clear the bar).

4. **Dashboard** (`app.py`) — a Streamlit app with:
   - A 3D hexagon map (PyDeck/H3) of historical or forecasted severity, log scaled for visibility across a long tailed severity distribution.
   - A predictive engine that forecasts severity for any selected hour/day combination.
   - A What-If Patrol Simulator — model the effect of deploying N patrol interceptors on citywide severity 
   - SHAP-based explainability — shows which features (hour, day, historical violation volume) drove a zone's severity score.
   - An LLM-generated deployment briefing (Gemini).


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
