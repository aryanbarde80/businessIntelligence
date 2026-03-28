"""Churn analysis helpers."""
from __future__ import annotations

import pandas as pd


def churn_summary(user_metrics: pd.DataFrame) -> pd.Series:
    churn_rate = user_metrics["churn_flag"].mean()
    top_countries = (
        user_metrics[user_metrics["churn_flag"]]
        .groupby("country")["user_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(3)
    )
    return pd.Series(
        {
            "global_churn_rate": round(churn_rate, 4),
            "top_churn_countries": ",".join(top_countries.index.tolist()),
        }
    )
