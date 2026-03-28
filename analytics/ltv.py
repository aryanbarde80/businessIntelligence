"""Lifetime value and revenue quality helpers."""
from __future__ import annotations

import pandas as pd


def ltv_summary(user_metrics: pd.DataFrame) -> pd.Series:
    if user_metrics.empty:
        return pd.Series(
            {
                "arpu": 0.0,
                "paid_arpu": 0.0,
                "estimated_ltv": 0.0,
                "renewal_rate_mean": 0.0,
            }
        )

    churn_rate = max(user_metrics["churn_flag"].mean(), 1e-6)
    arpu = user_metrics["payments_total"].mean()
    paid = user_metrics[user_metrics["is_paid_user"]]
    paid_arpu = paid["payments_total"].mean() if not paid.empty else 0.0

    # Simple heuristic LTV: ARPU divided by churn; capped for stability
    est_ltv = min(arpu / churn_rate, arpu * 24)
    renewal_rate_mean = user_metrics["renewal_rate"].mean()

    return pd.Series(
        {
            "arpu": round(arpu, 2),
            "paid_arpu": round(paid_arpu, 2),
            "estimated_ltv": round(est_ltv, 2),
            "renewal_rate_mean": round(renewal_rate_mean, 3),
        }
    )
