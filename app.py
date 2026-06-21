import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import shap
from streamlit_shap import st_shap
import numpy as np
import os
from google import genai
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

_gemini_client = genai.Client() if os.getenv("GEMINI_API_KEY") else None

FEATURE_COLS = ['hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 'total_violations']

# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_briefing(zone_label, hour, severity_score, tier_label):
    if _gemini_client is None:
        return (f"⚠️ LLM briefing unavailable (no API key configured). "
                f"Severity score: {severity_score:.1f} ({tier_label}).")
    prompt = (
        f"You are a traffic enforcement operations assistant. Write a 2-sentence "
        f"deployment briefing for a patrol supervisor.\n\n"
        f"Zone: {zone_label}\nTime: {hour}:00\n"
        f"Congestion severity score: {severity_score:.1f} ({tier_label})\n\n"
        f"Be concrete and operational. Do not invent statistics beyond what's given."
    )
    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        return response.text
    except Exception as e:
        err_str = str(e)
        if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
            return (f"⚠️ Daily API quota reached. Severity score: {severity_score:.1f} "
                    f"({tier_label}) — recommend deployment to zone {zone_label}.")
        return f"⚠️ Briefing generation failed. Severity score: {severity_score:.1f} ({tier_label})."


def get_statute_reference(severity_score, df_all):
    p90 = df_all['aggregated_severity'].quantile(0.90)
    p75 = df_all['aggregated_severity'].quantile(0.75)
    if severity_score >= p90:
        return "CRITICAL (Tier 1)", "MV Act, 1988 — Section 122 / 177: Obstructive hazard — towing authorized."
    elif severity_score >= p75:
        return "WARNING (Tier 2)", "MV Act, 1988 — Section 201: Obstruction to free flow of traffic."
    else:
        return "STANDARD (Tier 3)", "MV Act, 1988 — Section 177: General minor parking/traffic infraction."


def predict_zones(df, selected_hour, selected_dow, is_weekend_sel, is_peak_sel,
                  violation_multiplier=1.0):
    """
    Scores only the h3_cells that are historically active at this exact hour+day.
    This keeps the zone count identical to the historical view — only severity changes.
    Clips predictions to >= 0 so log-scaling and the map stay well-behaved.
    """
    # ── KEY FIX: restrict to zones active at this specific hour/day ──
    hour_day_df = df[
        (df['hour'] == selected_hour) & (df['day_of_week'] == selected_dow)
    ]

    zone_avg = hour_day_df.groupby('h3_cell').agg(
        total_violations=('total_violations', 'mean'),
        mean_latitude=('mean_latitude', 'mean'),
        mean_longitude=('mean_longitude', 'mean')
    ).reset_index()

    zone_avg['hour'] = selected_hour
    zone_avg['day_of_week'] = selected_dow
    zone_avg['is_weekend'] = is_weekend_sel
    zone_avg['is_peak_hour'] = is_peak_sel
    zone_avg['total_violations'] = zone_avg['total_violations'] * violation_multiplier

    preds = model.predict(zone_avg[FEATURE_COLS])
    zone_avg['predicted_severity'] = np.clip(preds, 0, None)

    return zone_avg.sort_values('predicted_severity', ascending=False)


def build_hex_layer(data: pd.DataFrame, weight_col: str) -> pdk.Layer:
    df = data.copy()
    df['color_weight_log'] = np.log1p(df[weight_col])
    return pdk.Layer(
        "HexagonLayer",
        df,
        get_position=["mean_longitude", "mean_latitude"],
        get_weight="color_weight_log",
        elevation_aggregation="SUM",
        radius=180,
        coverage=0.85,
        elevation_scale=40,
        extruded=True,
        pickable=True,
        color_range=[
            [33, 102, 172],
            [255, 247, 188],
            [254, 178, 76],
            [240, 59, 32],
            [128, 0, 38],
        ],
    )


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="AI Traffic Intelligence Platform")


@st.cache_data
def load_dashboard_data():
    return pd.read_csv("data/master_table.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/ensemble_forecaster.pkl")


df = load_dashboard_data()
model = load_model()

st.title("AI-Driven Traffic Intelligence System")
st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────

ctrl_col, sim_col = st.columns([1, 1])

with ctrl_col:
    st.header("⚙️ Operations Room")
    selected_hour = st.slider("Target Analysis Hour", 0, 23, 10)
    selected_day = st.selectbox(
        "Day of Week Analysis",
        ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]
    )
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
               "Friday": 4, "Saturday": 5, "Sunday": 6}
    selected_dow = day_map[selected_day]
    is_weekend_sel = 1 if selected_dow >= 5 else 0
    is_peak_sel = 1 if (8 <= selected_hour <= 12) or (16 <= selected_hour <= 20) else 0

    if st.button("🔮 Run Predictive Engine"):
        with st.spinner("Scoring zones…"):
            zones = predict_zones(df, selected_hour, selected_dow,
                                  is_weekend_sel, is_peak_sel)
            st.session_state['predicted_zones'] = zones
            st.session_state['predicted_for'] = (selected_hour, selected_dow)
            # Clear any stale simulation state
            st.session_state.pop('sim_baseline', None)
            st.session_state.pop('sim_result', None)
            st.session_state.pop('sim_interceptors', None)
        st.success(
            f"Forecast ready — {selected_day} {selected_hour}:00 "
            f"across {len(zones)} zones."
        )

# ── What-If Patrol Simulator ──────────────────────────────────────────────────

with sim_col:
    st.header("🚔 What-If Patrol Simulator")
    st.caption(
        "Deploy interceptors and re-score the city. "
        "Watch critical zones shrink and cool on the map in real time."
    )

    interceptors = st.slider("Deploy Interceptors", 0, 10, 0,
                             help="Each unit suppresses violation load by ~6%")
    reduction_factor = max(0.0, 1.0 - interceptors * 0.06)

    if interceptors > 0:
        st.info(
            f"**{interceptors} interceptor{'s' if interceptors > 1 else ''} deployed** — "
            f"violation load reduced by **{(1 - reduction_factor) * 100:.0f}%**"
        )

    sim_disabled = interceptors == 0
    if st.button("🔁 Simulate Patrol Deployment", disabled=sim_disabled):
        with st.spinner("Re-scoring under patrol conditions…"):
            # Baseline (no patrol) for delta comparison
            baseline_zones = predict_zones(df, selected_hour, selected_dow,
                                           is_weekend_sel, is_peak_sel,
                                           violation_multiplier=1.0)
            baseline_severity = baseline_zones['predicted_severity'].sum()

            # Simulated (patrol suppression)
            sim_zones = predict_zones(df, selected_hour, selected_dow,
                                      is_weekend_sel, is_peak_sel,
                                      violation_multiplier=reduction_factor)
            simulated_severity = sim_zones['predicted_severity'].sum()

            reduction_pct = (
                (baseline_severity - simulated_severity) / baseline_severity * 100
                if baseline_severity > 0 else 0
            )

            st.session_state['predicted_zones'] = sim_zones
            st.session_state['predicted_for'] = (selected_hour, selected_dow)
            st.session_state['sim_baseline'] = baseline_severity
            st.session_state['sim_result'] = simulated_severity
            st.session_state['sim_interceptors'] = interceptors

        st.success(
            f"Citywide severity reduced by **{reduction_pct:.1f}%** "
            f"under {interceptors}-unit deployment."
        )

    if 'sim_baseline' in st.session_state:
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Severity", f"{st.session_state['sim_baseline']:.0f}")
        m2.metric(
            "Post-Patrol Severity",
            f"{st.session_state['sim_result']:.0f}",
            delta=f"-{st.session_state['sim_baseline'] - st.session_state['sim_result']:.0f}",
            delta_color="inverse"
        )
        m3.metric("Interceptors", st.session_state['sim_interceptors'])

# ── Map ───────────────────────────────────────────────────────────────────────

hour_df = df[
    (df['hour'] == selected_hour) & (df['day_of_week'] == selected_dow)
].copy()

stale = st.session_state.get('predicted_for') != (selected_hour, selected_dow)
use_prediction = ('predicted_zones' in st.session_state) and not stale

if use_prediction:
    map_data = st.session_state['predicted_zones']
    weight_col = "predicted_severity"
else:
    map_data = hour_df
    weight_col = "aggregated_severity"

st.subheader("Micro-Hotspot Spatial Severity Topography")

view_state = pdk.ViewState(
    longitude=hour_df['mean_longitude'].mean() if not hour_df.empty else 77.5946,
    latitude=hour_df['mean_latitude'].mean() if not hour_df.empty else 12.9716,
    zoom=11.5,
    pitch=50,
    bearing=-25,
)

sim_active = 'sim_baseline' in st.session_state and use_prediction

st.pydeck_chart(pdk.Deck(
    layers=[build_hex_layer(map_data, weight_col)],
    initial_view_state=view_state,
    map_style="dark",
    tooltip={
        "html": (
            "<b>Severity:</b> {elevationValue}<br/>"
            "<b>Log-scaled color:</b> {colorValue}"
        ),
        "style": {"backgroundColor": "#222", "color": "white", "fontSize": "13px"}
    }
))

st.caption(
    "🔵 Low → 🟡 Moderate → 🟠 High → 🔴 Critical. "
    "Column height = severity score. Color uses log scale for mid-range visibility."
    + (" 🚔 *Patrol simulation active — map reflects suppressed severity.*"
       if sim_active else "")
)

st.markdown("---")

# ── SHAP Explainability ───────────────────────────────────────────────────────

st.header("AI Explainability (SHAP)")
st.info("Understanding why the model flagged this specific spatial zone.")
 
if not hour_df.empty:
    worst_cell_features = hour_df.loc[[hour_df['aggregated_severity'].idxmax()]]
    h3_id = worst_cell_features.iloc[0]['h3_cell']
    X_explain = worst_cell_features[FEATURE_COLS]
 
    st.caption(f"Explaining zone: `{h3_id}` — {selected_day} {selected_hour}:00")
 
    try:
        if hasattr(model, 'named_estimators_'):
            explainer = shap.TreeExplainer(model.named_estimators_['lgb'])
        else:
            explainer = shap.TreeExplainer(model)

        shap_matrix = explainer(X_explain)
 
        clean_explanation = shap.Explanation(
            values=shap_matrix.values.flatten(),
            base_values=float(np.array(shap_matrix.base_values).flatten()[0]),
            data=shap_matrix.data.flatten(),
            feature_names=X_explain.columns.tolist()
        )
        st.write("### Local Feature Contribution Waterfall")
        shap.plots.waterfall(clean_explanation, show=False)
        fig = plt.gcf()
        st_shap(fig, height=320)
        plt.close(fig)
 
    except Exception as e:
        st.error(f"SHAP Engine error: {e}")
else:
    st.warning(
        f"No historical records for {selected_day} at {selected_hour}:00 "
        "— try a different hour/day combination."
    )
 
st.markdown("---")
 
# ── Prescriptive Operational Support ─────────────────────────────────────────

st.header("🤖 Prescriptive Operational Support")

worst_cell = None
if use_prediction:
    worst_cell = st.session_state['predicted_zones'].iloc[0]
    severity_for_alert = worst_cell['predicted_severity']
    source_label = "🔥 Live Forecast Model Output"
    if sim_active:
        source_label += " (patrol simulation active)"
elif not hour_df.empty:
    worst_cell = hour_df.loc[hour_df['aggregated_severity'].idxmax()]
    severity_for_alert = worst_cell['aggregated_severity']
    source_label = "⚠️ Historical Baseline — click 'Run Predictive Engine' to forecast"

if worst_cell is not None:
    tier_label, statute = get_statute_reference(severity_for_alert, df)

    copilot_col, rag_col = st.columns(2)

    with copilot_col:
        st.subheader("LLM Deployment Briefing")
        st.caption(source_label)
        briefing_text = generate_briefing(
            worst_cell['h3_cell'], selected_hour, severity_for_alert, tier_label
        )
        st.info(briefing_text)

    with rag_col:
        st.subheader("Statutory Reference Card")
        st.warning(
            f"**Tier:** {tier_label}\n\n"
            f"**Applicable section:** {statute}\n\n"
            f"_Indicative reference — verify against current state notification before action._"
        )
else:
    st.warning("No data available for this hour/day combination.")