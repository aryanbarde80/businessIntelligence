import sys
from pathlib import Path
from typing import Iterable

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analytics import cohorts, conversion, churn as churn_analysis, segmentation
from analytics.anomaly import detect_dau_anomalies
from analytics.ltv import ltv_summary
from backend.processor import run_processing
from data.generator import run_simulation
from insights import rules
from ml.churn_model import MODEL_DIR

import pandas as pd
import plotly.express as px
from datetime import datetime

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

st.sidebar.title("How to Use")
with st.sidebar.expander("Quick tour", expanded=False):
    st.markdown(
        """
        1) Pick **Countries** and **Plans** to focus the KPIs.
        2) Drag the **Session window** to explore different periods.
        3) Scroll down for funnel, cohorts, and churn hotspots.
        4) Use **Regenerate demo data** to refresh with a new synthetic cohort.
        """
    )

if st.sidebar.button("🔄 Regenerate demo data"):
    with st.spinner("Running full pipeline (generate → clean → score)…"):
        _ensure_processed_tables(["user_metrics", "sessions", "events"])
        run_simulation()
        run_processing()
        st.cache_data.clear()
    st.success("Pipeline complete. Tables and artifacts refreshed.")

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
if pd.isna(min_date):
    min_date = datetime.utcnow()
if pd.isna(max_date):
    max_date = min_date
min_date_dt = pd.to_datetime(min_date).to_pydatetime()
max_date_dt = pd.to_datetime(max_date).to_pydatetime()
start_date, end_date = st.sidebar.slider(
    "Session window", min_value=min_date_dt, max_value=max_date_dt, value=(min_date_dt, max_date_dt)
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

st.markdown("---")
st.subheader("Revenue & LTV")
ltv = ltv_summary(filtered_metrics)
rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
rev_col1.metric("ARPU (all)", f"${ltv['arpu']}")
rev_col2.metric("ARPU (paid)", f"${ltv['paid_arpu']}")
rev_col3.metric("Est. LTV", f"${ltv['estimated_ltv']}")
rev_col4.metric("Renewal rate", f"{ltv['renewal_rate_mean']:.2%}")

st.markdown("---")
st.subheader("Health: DAU Anomalies")
anomalies = detect_dau_anomalies(filtered_sessions)
if anomalies.empty:
    st.info("No session data yet.")
else:
    anom_chart = px.bar(
        anomalies,
        x="date",
        y="dau",
        color=anomalies["is_anomaly"].map({True: "Anomaly", False: "Normal"}),
        title="Daily Active Users with Anomaly Flags",
    )
    st.plotly_chart(anom_chart, use_container_width=True)
    st.dataframe(anomalies.tail(15))

st.markdown("---")
st.subheader("Churn Drivers")
fi_path = MODEL_DIR / "churn_feature_importance.csv"
if fi_path.exists():
    fi_df = pd.read_csv(fi_path)
    if "importance" in fi_df.columns:
        fi_df["feature"] = fi_df.get("Unnamed: 0", fi_df.index)
        fi_df = fi_df[["feature", "importance"]].head(8)
        fig_fi = px.bar(fi_df, x="importance", y="feature", orientation="h", title="Top churn drivers")
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance not available yet.")
else:
    st.info("Run the pipeline to compute churn feature importance.")

st.caption("Backend tip: set environment variable MODEL_BACKEND=xgboost to train gradient-boosted churn model; default is logistic regression.")
st.markdown("---")
st.subheader("Data Guide")
guide_col1, guide_col2 = st.columns(2)
guide_col1.markdown(
    """
    **Typical questions to ask**
    - Are paid users churning faster in certain countries?
    - Which features drive the most interactions?
    - How sticky are the latest signup cohorts?
    """
)
guide_col2.markdown(
    """
    **How to read the charts**
    - *Funnel*: tracks drop-off from signups → activity → paid → churned.
    - *Cohort heatmap*: darker cells = higher weekly retention for that signup cohort.
    - *DAU line*: spikes indicate feature launches or campaigns.
    """
)

st.caption("Tip: Use the sidebar filters and session window to create focused reviews, then regenerate demo data to test different scenarios.")
