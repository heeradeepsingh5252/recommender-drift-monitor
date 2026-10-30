"""Streamlit dashboard for the Recommender Drift Monitor API."""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

DEFAULT_API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 5

st.set_page_config(
    page_title="Recommender Drift Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_get(base_url: str, path: str) -> dict:
    response = requests.get(f"{base_url}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, payload: dict | None = None) -> dict:
    response = requests.post(
        f"{base_url}{path}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


st.title("📡 Recommender Drift Monitor")
st.caption("An end-to-end demo of personalized recommendations, population monitoring, and drift simulation.")

with st.sidebar:
    st.header("Control panel")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL).rstrip("/")
    user_id = st.number_input("User ID for recommendations", min_value=1, max_value=120, value=7, step=1)
    recommendation_count = st.slider("Number of recommendations", min_value=3, max_value=15, value=8)
    st.divider()
    st.subheader("Drift simulation")
    st.write("Use the simulated shift to see how genre and rating distributions change.")
    trigger_drift = st.button("⚠️ Simulate drift", use_container_width=True, type="primary")
    reset_drift = st.button("↩️ Reset to normal", use_container_width=True)
    st.button("🔄 Refresh dashboard", use_container_width=True)

if trigger_drift or reset_drift:
    try:
        if trigger_drift:
            result = api_post(api_url, "/drift/simulate", {"shifted": True})
        else:
            result = api_post(api_url, "/drift/reset")
        st.toast(f"Monitor status: {result.get('status', 'updated')}")
    except requests.RequestException as error:
        st.error(f"Could not update drift state: {error}")
        st.stop()

try:
    health = api_get(api_url, "/health")
    overview = api_get(api_url, "/overview")
    report = api_get(api_url, "/drift/report")
    recommendations_payload = api_get(
        api_url, f"/recommend/{int(user_id)}?limit={int(recommendation_count)}"
    )
except requests.RequestException as error:
    st.error("The API is not reachable yet.")
    st.markdown(
        "Start it from the project folder with `./run_local.sh`, or launch the API separately "
        "using `python -m uvicorn recommender_drift_monitor.api:app --app-dir src --reload`."
    )
    st.code(str(error))
    st.stop()

status = overview.get("status", "UNKNOWN")
metric_cols = st.columns(5)
metric_cols[0].metric("API status", health.get("status", "unknown").upper())
metric_cols[1].metric("Users", f"{overview.get('user_count', 0):,}")
metric_cols[2].metric("Catalog items", f"{overview.get('item_count', 0):,}")
metric_cols[3].metric("Current interactions", f"{overview.get('current_interactions', 0):,}")
metric_cols[4].metric("Drift status", status, delta="Review needed" if overview.get("drift_detected") else "Within threshold", delta_color="inverse" if overview.get("drift_detected") else "normal")

if overview.get("drift_detected"):
    st.warning(f"Distribution drift detected in: {', '.join(report.get('drifted_features', []))}")
else:
    st.success("No monitored feature currently exceeds the configured PSI threshold.")

st.subheader("Data profile: reference vs current")
left, right = st.columns(2)
with left:
    genre_data = pd.DataFrame(overview.get("genre_distribution", []))
    if not genre_data.empty:
        genre_long = genre_data.melt(
            id_vars="genre", value_vars=["reference", "current"],
            var_name="window", value_name="interactions",
        )
        fig_genre = px.bar(
            genre_long, x="genre", y="interactions", color="window", barmode="group",
            title="Genre distribution", labels={"genre": "Genre", "interactions": "Interactions", "window": "Dataset"},
        )
        fig_genre.update_layout(legend_title_text="Dataset", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_genre, use_container_width=True)
with right:
    rating_data = pd.DataFrame(overview.get("rating_distribution", []))
    if not rating_data.empty:
        rating_long = rating_data.melt(
            id_vars="rating", value_vars=["reference", "current"],
            var_name="window", value_name="interactions",
        )
        fig_rating = px.bar(
            rating_long, x="rating", y="interactions", color="window", barmode="group",
            title="Rating distribution", labels={"rating": "Rating", "interactions": "Interactions", "window": "Dataset"},
        )
        fig_rating.update_layout(legend_title_text="Dataset", margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig_rating, use_container_width=True)

st.subheader("Drift report")
drift_frame = pd.DataFrame(report.get("features", []))
if not drift_frame.empty:
    drift_frame["psi"] = drift_frame["psi"].astype(float)
    st.dataframe(
        drift_frame.rename(columns={
            "feature": "Feature", "psi": "PSI", "threshold": "Threshold",
            "drift_detected": "Drift detected", "reference_mean": "Reference mean",
            "current_mean": "Current mean",
        }),
        use_container_width=True,
        hide_index=True,
    )
st.caption("PSI thresholds are heuristics, not a guarantee of model quality or a substitute for domain-specific validation.")

st.subheader(f"Recommendations for user {int(user_id)}")
recommendations = pd.DataFrame(recommendations_payload.get("recommendations", []))
if not recommendations.empty:
    recommendations = recommendations.rename(columns={
        "item_id": "Item ID", "title": "Title", "genre": "Genre",
        "score": "Predicted score", "rating_count": "Training ratings",
    })
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
else:
    st.info("No recommendations were returned for this user.")

with st.expander("Project notes and limitations"):
    st.markdown(
        """
        - The catalog and interaction logs are synthetic and generated locally.
        - Recommendations use item-item collaborative filtering on the reference window.
        - Drift is monitored with PSI for genre and rating distributions.
        - The drift simulation is a controlled demonstration, not a production alerting system.
        - For production, validate thresholds, monitor online recommendation outcomes, add data-quality checks,
          and evaluate ranking quality with a held-out dataset.
        """
    )
