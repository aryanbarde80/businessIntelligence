"""Conversion rate helpers."""
from __future__ import annotations

import pandas as pd


def conversion_rates(user_metrics: pd.DataFrame) -> pd.Series:
    total = len(user_metrics)
    paid = user_metrics[user_metrics["is_paid_user"]]
    conversion = paid["user_id"].nunique() / max(total, 1)
    active = user_metrics[user_metrics["sessions_total"] > 5]
    active_conversion = paid["user_id"].nunique() / max(len(active), 1)
    return pd.Series(
        {
            "total_users": total,
            "paid_users": paid["user_id"].nunique(),
            "conversion_rate": round(conversion, 4),
            "active_conversion_rate": round(active_conversion, 4),
        }
    )
