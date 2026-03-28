"""Rule-based insight generation."""
from __future__ import annotations

from typing import List

import pandas as pd


def generate_insights(user_metrics: pd.DataFrame, events: pd.DataFrame) -> List[str]:
    insights: List[str] = []
    if user_metrics.empty:
        return insights

    churned = user_metrics[user_metrics["churn_flag"]]
    if not churned.empty:
        country = churned.groupby("country")["user_id"].nunique().idxmax()
        insights.append(f"Users from {country} show the highest churn within the cohort.")

    feature_usage = events.groupby("feature")["event_id"].count()
    if not feature_usage.empty:
        top_feature = feature_usage.idxmax()
        insights.append(f"Feature {top_feature} drives the most interactions and keeps users engaged.")

    plan_retention = (
        user_metrics.groupby("plans")["renewal_rate"].mean().sort_values(ascending=False)
    )
    if not plan_retention.empty:
        best_plan = plan_retention.idxmax()
        insights.append(f"Users on {best_plan} have the strongest renewal rate ({plan_retention.max():.2f}).")

    return insights
