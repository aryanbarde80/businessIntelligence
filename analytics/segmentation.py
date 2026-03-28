"""Segmentation summaries by country and plan."""
from __future__ import annotations

import pandas as pd


def segment_by_country(user_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        user_metrics.groupby("country")
        .agg(
            total_users=("user_id", "count"),
            avg_sessions=("sessions_total", "mean"),
            avg_revenue=("payments_total", "mean"),
            churn_rate=("churn_flag", "mean"),
        )
        .assign(churn_rate=lambda df: df["churn_rate"].round(3))
        .sort_values("total_users", ascending=False)
    )
    return summary


def segment_by_plan(user_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        user_metrics.groupby("plans")
        .agg(
            total_users=("user_id", "count"),
            avg_payment=("payments_total", "mean"),
            avg_sessions=("sessions_total", "mean"),
            churn_rate=("churn_flag", "mean"),
        )
        .assign(churn_rate=lambda df: df["churn_rate"].round(3))
        .sort_values("total_users", ascending=False)
    )
