"""ML feature engineering helpers."""
from __future__ import annotations

from typing import List

import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "sessions_total",
    "avg_session_duration",
    "session_span_days",
    "payments_total",
    "payment_count",
    "renewal_rate",
]


def prepare_ml_features(user_metrics: pd.DataFrame) -> pd.DataFrame:
    df = user_metrics[FEATURE_COLUMNS + ["churn_flag"]].copy()
    df["churn_flag"] = df["churn_flag"].astype(int)
    df.fillna(0, inplace=True)
    return df
