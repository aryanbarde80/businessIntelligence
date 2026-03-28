"""Revenue and retention helpers."""
from __future__ import annotations

import pandas as pd


def monthly_revenue(payments: pd.DataFrame) -> pd.DataFrame:
    if payments.empty:
        return pd.DataFrame(columns=["month", "revenue"])
    df = payments.copy()
    df["month"] = df["payment_date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby("month")["revenue"]
        .sum()
        .reset_index()
        .sort_values("month")
    )
    return monthly


def revenue_summary(payments: pd.DataFrame) -> pd.Series:
    monthly = monthly_revenue(payments)
    if monthly.empty:
        return pd.Series(
            {
                "current_mrr": 0.0,
                "prev_mrr": 0.0,
                "mrr_growth": 0.0,
            }
        )
    current = monthly.iloc[-1]["revenue"]
    prev = monthly.iloc[-2]["revenue"] if len(monthly) > 1 else current
    growth = ((current - prev) / prev) if prev else 0.0
    return pd.Series(
        {
            "current_mrr": round(current, 2),
            "prev_mrr": round(prev, 2),
            "mrr_growth": round(growth, 4),
        }
    )
