import sys
from pathlib import Path
from typing import Iterable

import streamlit as st
import joblib
import shap

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analytics import cohorts, conversion, churn as churn_analysis, segmentation
from analytics.anomaly import detect_dau_anomalies
from analytics.ltv import ltv_summary
from analytics.forecast import forecast_dau
from analytics.revenue import monthly_revenue, revenue_summary
from backend.processor import run_processing
from data.generator import run_simulation
from insights import rules
from ml.churn_model import MODEL_DIR
from ml.features import FEATURE_COLUMNS

import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="SaaS Behavior Dashboard", layout="wide")

# Theme toggle
theme_choice = st.sidebar.radio("Theme", ["Midnight", "Ivory"], index=0, horizontal=True)
is_dark = theme_choice == "Midnight"
accent_presets = {
    "Electric": "#7bd5f5",
    "Emerald": "#10b981",
    "Sunset": "#f97316",
}
preset_choice = st.sidebar.selectbox("Accent preset", ["Electric", "Emerald", "Sunset", "Custom"], index=0)
accent_color = st.sidebar.color_picker("Custom accent", value="#7bd5f5" if is_dark else "#2563eb")
if preset_choice != "Custom":
    accent_color = accent_presets[preset_choice]
chart_template = st.sidebar.selectbox("Chart style", ["plotly_dark", "plotly_white"], index=0 if is_dark else 1)

# simple route state
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def go(page: str):
    st.session_state["page"] = page

# Custom styling for a modern, app-like feel with theme variables
st.markdown(
    f"""
    <style>
    :root {{
      --bg: {"#070b12" if is_dark else "#f7f8fb"};
      --card: {"#0f182b" if is_dark else "#ffffff"};
      --card2: {"#10192f" if is_dark else "#f0f4ff"};
      --border: {"#1f2a44" if is_dark else "#e4e7ef"};
      --text: {"white" if is_dark else "#0f172a"};
      --accent: {accent_color};
    }}
    body, .block-container {{background: var(--bg);}}
    .metric-card {{background: linear-gradient(135deg, var(--card), var(--card2)); padding:16px 18px; border-radius:14px; color:var(--text); border:1px solid var(--border);}}
    .metric-label {{font-size:13px; opacity:0.8;}}
    .metric-value {{font-size:22px; font-weight:700;}}
    .pill {{display:inline-block; padding:4px 10px; border-radius:999px; background:var(--card); color:var(--accent); border:1px solid var(--border); font-size:12px;}}
    .glass {{backdrop-filter: blur(12px); background:var(--card); border:1px solid var(--border); border-radius:16px; padding:18px;}}
    .section-title {{font-size:20px; font-weight:700; margin-bottom:8px; color:var(--text);}}
    /* Navbar */
    .top-nav {{position:sticky; top:0; z-index:50; background:var(--card); border-bottom:1px solid var(--border); padding:12px 24px; display:flex; align-items:center; gap:18px;}}
    .nav-brand {{font-weight:800; color:var(--accent); font-size:18px;}}
    .nav-link {{color:var(--text); text-decoration:none; font-weight:600; padding:6px 10px; border-radius:10px;}}
    .nav-link:hover {{background:var(--border);}}
    .btn {{padding:12px 18px; border-radius:14px; border:none; font-weight:700; cursor:pointer; background:var(--accent); color:white;}}
    .btn-ghost {{padding:12px 18px; border-radius:14px; border:1px solid var(--border); background:transparent; color:var(--text); font-weight:700;}}
    /* Footer */
    .footer {{margin-top:24px; padding:16px; text-align:center; color:var(--text); opacity:0.8; border-top:1px solid var(--border);}}
    </style>
    """,
    unsafe_allow_html=True,
)

# Navbar
st.markdown(
    """
    <div class="top-nav">
      <span class="nav-brand">Nimbus Analytics</span>
      <a class="nav-link" href="#" onclick="window.parent.postMessage({type:'streamlit:setPage','page':'home'}, '*')">Home</a>
      <a class="nav-link" href="#kpis">Analytics</a>
      <a class="nav-link" href="#revenue">Revenue</a>
      <a class="nav-link" href="#health">Health</a>
      <a class="nav-link" href="#churn">Churn</a>
      <a class="nav-link" href="#forecast">Forecast</a>
      <a class="nav-link" href="#downloads">Downloads</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Route selector (web-like tabs)
nav_choice = st.radio(
    "Navigation",
    ["Home", "Analytics"],
    horizontal=True,
    index=0 if st.session_state["page"] == "home" else 1,
)
st.session_state["page"] = "home" if nav_choice == "Home" else "analytics"

def render_home():
    # Hero section inspired by devin.ai style
    st.markdown(
        f"""
        <div style="margin-top:8px; padding:32px; border-radius:28px; background:radial-gradient(circle at 20% 20%, {accent_color}33, transparent 35%), radial-gradient(circle at 80% 0%, {accent_color}22, transparent 30%), linear-gradient(135deg, #0f111a, #0c1625); border:1px solid var(--border); color:var(--text);">
          <div style="display:flex; gap:32px; flex-wrap:wrap; align-items:center;">
            <div style="flex:1; min-width:280px;">
              <div style="font-size:42px; font-weight:800; line-height:1.1;">Ship analytics like an AI engineer.</div>
              <div style="font-size:17px; opacity:0.85; margin-top:10px;">
                Auto-generate data, pipeline it, score churn, forecast demand, and explain decisions — all inside one Streamlit experience.
              </div>
              <div style="margin-top:18px; display:flex; gap:12px; flex-wrap:wrap;">
                <button class="btn" onClick="window.location.reload()">Regenerate demo data</button>
                <button class="btn-ghost" onClick="window.location='#kpis'">Launch analytics</button>
              </div>
              <div style="margin-top:12px; display:flex; gap:12px; flex-wrap:wrap; opacity:0.85;">
                <span class="pill">Churn ML</span><span class="pill">DAU forecast</span><span class="pill">LTV + MRR</span><span class="pill">SHAP explainability</span>
              </div>
            </div>
            <div style="flex:1; min-width:260px; padding:16px; border-radius:18px; background:var(--card); border:1px solid var(--border);">
              <div style="font-weight:700; margin-bottom:10px;">Live preview snippet</div>
              <pre style="background:rgba(255,255,255,0.04); padding:14px; border-radius:12px; border:1px solid var(--border); color:var(--text); overflow:auto; font-size:12px;">
from run_pipeline import run_pipeline
run_pipeline()

# launch dashboard
streamlit run dashboard/app.py --server.port 8501
              </pre>
              <div style="font-size:12px; opacity:0.7;">One command to regenerate everything.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Why teams use Nimbus")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**End-to-end**  \nSimulate → ETL → DB → analytics → ML → insights.")
    c2.markdown("**Operator-friendly**  \nOne-click regenerate, themed UI, filters, downloads.")
    c3.markdown("**Explainable**  \nFeature importance + per-user SHAP waterfalls.")
    st.markdown("### Customization")
    st.markdown("- Choose accent preset or custom color; toggle Midnight/Ivory; switch chart template.\n- Filter by countries, plans, and session window.\n- Scenario planner to model conversion uplift and churn reduction.")
    st.info("Ready? Switch to the Analytics tab above or click Launch analytics.")

if st.session_state["page"] == "home":
    render_home()
    st.stop()

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

@st.cache_data
def load_payments() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "payments.csv", parse_dates=["payment_date"])
    return df

@st.cache_data
def load_scored() -> pd.DataFrame:
    path = PROCESSED_DIR.parent / "artifacts" / "churn_scored.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource
def load_churn_pipeline():
    path = MODEL_DIR / "churn_pipeline.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

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
payments = load_payments()
scored = load_scored()
churn_pipeline = load_churn_pipeline()

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

kpi_anchor = st.markdown("<div id='kpis'></div>", unsafe_allow_html=True)
kpi_columns = st.columns(4)
metrics = [
    ("Global Users", f"{int(kpi['total_users']):,}"),
    ("Paying Users", f"{int(kpi['paid_users']):,}"),
    ("Conversion", f"{float(kpi['conversion_rate']):.2%}"),
    ("Churn", f"{float(churn_kpi['global_churn_rate']):.2%}"),
]
for col, (label, val) in zip(kpi_columns, metrics):
    with col:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

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
trend_col1.plotly_chart(px.area(daily, x="date", y="dau", title="DAU trend", template=chart_template), use_container_width=True)

trend_col2.subheader("Plan Segments")
trend_col2.dataframe(segmentation_df.head(6))

st.subheader("Insights")
if insight_list:
    for insight in insight_list:
        st.markdown(f"- {insight}")
else:
    st.info("Insights will appear after pipeline reruns.")

st.markdown("---")
st.markdown("<div id='revenue'></div>", unsafe_allow_html=True)
st.subheader("Revenue & LTV")
ltv = ltv_summary(filtered_metrics)
rev_col1, rev_col2, rev_col3, rev_col4 = st.columns(4)
rev_col1.metric("ARPU (all)", f"${ltv['arpu']}")
rev_col2.metric("ARPU (paid)", f"${ltv['paid_arpu']}")
rev_col3.metric("Est. LTV", f"${ltv['estimated_ltv']}")
rev_col4.metric("Renewal rate", f"{ltv['renewal_rate_mean']:.2%}")

rev_summary = revenue_summary(payments[payments["user_id"].isin(filtered_metrics["user_id"])])
rev_col1, rev_col2, rev_col3 = st.columns(3)
rev_col1.metric("Current MRR", f"${rev_summary['current_mrr']}")
rev_col2.metric("Prev. MRR", f"${rev_summary['prev_mrr']}")
rev_col3.metric("MRR growth", f"{rev_summary['mrr_growth']:.2%}")

monthly_rev = monthly_revenue(payments[payments["user_id"].isin(filtered_metrics["user_id"])])
if not monthly_rev.empty:
    rev_chart = px.line(monthly_rev, x="month", y="revenue", title="Monthly Recurring Revenue")
    st.plotly_chart(rev_chart, use_container_width=True)

st.subheader("Scenario Planner")
uplift = st.slider("Conversion uplift (%)", min_value=0, max_value=50, value=10, step=5)
churn_cut = st.slider("Churn reduction (%)", min_value=0, max_value=50, value=5, step=5)
base_mrr = rev_summary["current_mrr"]
projected_mrr = base_mrr * (1 + uplift / 100) * (1 + churn_cut / 200)
delta = projected_mrr - base_mrr
st.metric("Projected MRR", f"${projected_mrr:,.0f}", delta=f"${delta:,.0f}")
st.caption("Approximation assumes uplift improves acquisition and half of churn reduction compounds retention.")

st.markdown("---")
st.markdown("<div id='health'></div>", unsafe_allow_html=True)
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
st.markdown("<div id='churn'></div>", unsafe_allow_html=True)
st.subheader("Churn Drivers")
fi_path = MODEL_DIR / "churn_feature_importance.csv"
if fi_path.exists():
    fi_df = pd.read_csv(fi_path)
    if "model_importance" in fi_df.columns or "permutation_importance" in fi_df.columns:
        if "Unnamed: 0" in fi_df.columns:
            fi_df = fi_df.rename(columns={"Unnamed: 0": "feature"})
        fi_df = fi_df[["feature"] + [c for c in fi_df.columns if c != "feature"]]
        fi_df = fi_df.fillna(0)
        top = fi_df.sort_values(by=fi_df.columns[1], key=abs, ascending=False).head(8)
    fig_fi = px.bar(top, x=top.columns[1], y="feature", orientation="h", title="Top churn drivers (model importance)", template=chart_template)
    st.plotly_chart(fig_fi, use_container_width=True)
    if "permutation_importance" in fi_df.columns:
        fig_perm = px.bar(top, x="permutation_importance", y="feature", orientation="h", title="Permutation importance (robust)", template=chart_template)
        st.plotly_chart(fig_perm, use_container_width=True)
    else:
        st.info("Feature importance not available yet.")
else:
    st.info("Run the pipeline to compute churn feature importance.")

st.caption("Backend tip: set environment variable MODEL_BACKEND=xgboost to train gradient-boosted churn model; default is logistic regression.")
st.markdown("---")
st.subheader("Churn Risk Leaderboard")
if scored.empty:
    st.info("Run the pipeline to generate churn scores.")
else:
    risk = scored.sort_values("churn_probability", ascending=False).head(15)
    fig_risk = px.bar(
        risk,
        x="churn_probability",
        y="user_id",
        orientation="h",
        title="Top 15 at-risk users",
        labels={"churn_probability": "Churn probability", "user_id": "User"},
        color="churn_probability",
        color_continuous_scale="Reds",
        template=chart_template,
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    st.dataframe(risk[["user_id", "churn_probability", "plans", "payments_total", "sessions_total"]])

st.subheader("Explain a User (SHAP)")
if scored.empty or churn_pipeline is None:
    st.info("Run the pipeline to enable SHAP explanations.")
else:
    explain_user = st.selectbox("Select user to explain", options=scored.sort_values("churn_probability", ascending=False)["user_id"].head(50))
    sample = scored[FEATURE_COLUMNS + ["user_id", "churn_probability"]].copy() if "FEATURE_COLUMNS" in globals() else scored
    sample = sample.dropna()
    target_row = sample[sample["user_id"] == explain_user]
    if not target_row.empty:
        model = churn_pipeline["model"]
        scaler = churn_pipeline.get("scaler")
        X = sample[[c for c in sample.columns if c in model.feature_names_in_.tolist()]] if hasattr(model, "feature_names_in_") else sample.drop(columns=["user_id", "churn_probability"], errors="ignore")
        x_row = X[X.index == target_row.index[0]]
        try:
            if scaler is not None:
                X_use = scaler.transform(X)
                x_explain = scaler.transform(x_row)
            else:
                X_use = X
                x_explain = x_row
            explainer = shap.Explainer(model, X_use)
            shap_values = explainer(x_explain)
            st.write(f"Churn probability: {float(target_row['churn_probability'].iloc[0]):.2%}")
            st.pyplot(shap.plots.waterfall(shap_values[0], show=False), clear_figure=True)
        except Exception as exc:
            st.info(f"Could not compute SHAP for this model: {exc}")

st.markdown("<div id='forecast'></div>", unsafe_allow_html=True)
st.subheader("Forecast: DAU (14-day)")
forecast_df = forecast_dau(filtered_sessions)
if forecast_df.empty:
    st.info("Not enough history to forecast yet. Need at least 7 days.")
else:
    fc_fig = px.line(
        forecast_df,
        x="date",
        y=["dau", "forecast"],
        title="Historical DAU and 14-day forecast",
        labels={"value": "DAU", "variable": "Series"},
    )
    st.plotly_chart(fc_fig, use_container_width=True)
    st.dataframe(forecast_df.tail(10))

st.caption("Forecast uses simple exponential smoothing; adjust by setting MODEL_BACKEND or swapping forecaster in analytics/forecast.py.")
st.markdown("---")
st.markdown("<div id='downloads'></div>", unsafe_allow_html=True)
st.subheader("Downloads")
summary_path = ROOT_DIR / "artifacts" / "analytics_summary.json"
if summary_path.exists():
    with summary_path.open("r", encoding="utf-8") as fh:
        summary_payload = fh.read()
    st.download_button("Download analytics_summary.json", data=summary_payload, file_name="analytics_summary.json")
else:
    st.info("Run pipeline to generate summary artifacts.")

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
st.markdown("<div class='footer'>Built with Streamlit · Nimbus Analytics · Modern SaaS intelligence</div>", unsafe_allow_html=True)
