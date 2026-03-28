"""Simple anomaly detection for activity metrics."""
from __future__ import annotations

import pandas as pd


def detect_dau_anomalies(sessions: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """Return daily active users with z-scores and anomaly flags."""
    if sessions.empty or "session_time" not in sessions:
        return pd.DataFrame(columns=["date", "dau", "z_score", "is_anomaly"])

    daily = (
        sessions.dropna(subset=["session_time"])
        .groupby(sessions["session_time"].dt.date)["user_id"]
        .nunique()
        .reset_index(name="dau")
    )
    if daily.empty:
        return pd.DataFrame(columns=["date", "dau", "z_score", "is_anomaly"])

    mean = daily["dau"].mean()
    std = daily["dau"].std(ddof=0) or 1e-6
    daily["z_score"] = (daily["dau"] - mean) / std
    daily["is_anomaly"] = daily["z_score"].abs() >= z_threshold
    return daily
