import sys
from pathlib import Path
from typing import Iterable

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analytics import cohorts, conversion, churn as churn_analysis, segmentation
from backend.processor import run_processing
from data.generator import run_simulation
from insights import rules

import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SaaS Behavior Dashboard", layout="wide")

PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def _ensure_processed_tables(names: Iterable[str]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    key_file = PROCESSED_DIR / "user_metrics.csv"
    if key_file.exists():
        return
    run_simulation()
    run_processing()


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    return pd.read_csv(path, parse_dates=True)

@st.cache_data
def load_sessions() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "sessions.csv", parse_dates=["session_time"])
    return df

@st.cache_data
def load_events() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "events.csv", parse_dates=["event_time"])
    return df

_ensure_processed_tables(["user_metrics", "sessions", "events"])

user_metrics = load_table("user_metrics")
sessions = load_sessions()
events = load_events()

countries = sorted(user_metrics["country"].dropna().unique())
plans = sorted(user_metrics["plans"].dropna().unique())

st.sidebar.header("Filters")
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries[:3])
selected_plans = st.sidebar.multiselect("Plans", plans, default=plans)

min_date = sessions["session_time"].min()
max_date = sessions["session_time"].max()
start_date, end_date = st.sidebar.slider(
    "Session window", min_value=min_date, max_value=max_date, value=(min_date, max_date)
)

filtered_metrics = user_metrics.copy()
if selected_countries:
    filtered_metrics = filtered_metrics[filtered_metrics["country"].isin(selected_countries)]
if selected_plans:
    filtered_metrics = filtered_metrics[filtered_metrics["plans"].isin(selected_plans)]

filtered_sessions = sessions[
    (sessions["session_time"] >= start_date) & (sessions["session_time"] <= end_date)
]
filtered_sessions = filtered_sessions[filtered_sessions["user_id"].isin(filtered_metrics["user_id"])]

kpi = conversion.conversion_rates(filtered_metrics)
churn_kpi = churn_analysis.churn_summary(filtered_metrics)
segmentation_df = segmentation.segment_by_country(filtered_metrics)
cohort_retention = cohorts.build_weekly_cohort(filtered_sessions)
insight_list = rules.generate_insights(filtered_metrics, events)

kpi_columns = st.columns(4)
kpi_columns[0].metric("Global Users", int(kpi["total_users"]))
kpi_columns[1].metric("Paying Users", int(kpi["paid_users"]))
conversion_rate = float(kpi["conversion_rate"])
churn_rate = float(churn_kpi["global_churn_rate"])
kpi_columns[2].metric("Conversion", f"{conversion_rate:.2%}")
kpi_columns[3].metric("Churn", f"{churn_rate:.2%}")

st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

funnel_df = pd.DataFrame(
    {
        "Stage": ["Signups", "Active (30d)", "Paid", "Churned"],
        "Count": [
            len(filtered_metrics),
            filtered_sessions["user_id"].nunique(),
            filtered_metrics[filtered_metrics["is_paid_user"]]["user_id"].nunique(),
            filtered_metrics[filtered_metrics["churn_flag"]]["user_id"].nunique(),
        ],
    }
)

chart_col1.subheader("Funnel")
chart_col1.plotly_chart(px.bar(funnel_df, x="Stage", y="Count", color="Stage", text="Count"), use_container_width=True)

chart_col2.subheader("Cohort Retention")
if not cohort_retention.empty:
    chart_col2.plotly_chart(
        px.imshow(cohort_retention, labels={"x": "Weeks from signup", "y": "Cohort week", "color": "Retention"}),
        use_container_width=True,
    )
else:
    chart_col2.info("Not enough data for cohort heatmap yet.")

trend_col1, trend_col2 = st.columns(2)
trend_col1.subheader("Daily Active Users")
daily = (
    filtered_sessions.groupby(filtered_sessions["session_time"].dt.date)["user_id"].nunique().reset_index()
)
daily.columns = ["date", "dau"]
trend_col1.line_chart(daily.set_index("date"))

trend_col2.subheader("Plan Segments")
trend_col2.dataframe(segmentation_df.head(6))

st.subheader("Insights")
if insight_list:
    for insight in insight_list:
        st.markdown(f"- {insight}")
else:
    st.info("Insights will appear after pipeline reruns.")
