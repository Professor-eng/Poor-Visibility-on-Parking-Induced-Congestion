# AI-Driven Parking & Congestion Intelligence

An AI based solution for identifying illegal parking hotspots in Bengaluru and gauging their effects on traffic congestion.

**Live demo:** https://poor-visibility-on-parking-induced-congestion-rydynvc8bnzzvcmb.streamlit.app/

---

## The Problem

On-street illegal parking and spillover parking near commercial areas, metro stations, and events choke carriageways and intersections.
**Why It’s Hard Today**
- Enforcement is patrol-based and reactive.
- No heatmap of parking violations vs. congestion impact.
- Difficult to prioritize enforcement zones.

## Approach

1. **Data pipeline** (`src/pipeline.py`) — cleans raw violation records, only keeps approved records, and assigns a severity score for each violation type (e.g. double-parking and road crossing obstruction are more severe than footpath parking which is a more direct physical chokepoint). Uber H3 spatial hexagons are used for locations.

2. **Feature engineering** (`src/features.py`) — combines violation data into a single row per `(H3 cell, hour, day-of-week)`, includes total violations, aggregated severity and mean coordinates.


3. **Forecasting model** (`src/train.py`) — trains a single LightGBM regressor and an ensemble of two or more models — LightGBM and XGBoost — automatically choosing to ship the ensemble if it gets a better score compared to the solo regressor, and the score is better than a meaningful margin (current run ships LightGBM alone, since the ensemble failed to clear the bar).

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

Run the data pipeline once:
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
