"""Cohort analysis helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_weekly_cohort(sessions: pd.DataFrame, lookback_weeks: int = 12) -> pd.DataFrame:
    """Return retention rates per cohort week for the specified window."""
    sessions = sessions.dropna(subset=["session_time"]).copy()
    sessions["session_week"] = sessions["session_time"].dt.to_period("W").apply(lambda p: p.start_time)
    first_weeks = sessions.groupby("user_id")["session_week"].min()
    sessions["cohort_week"] = sessions["user_id"].map(first_weeks)
    sessions = sessions.dropna(subset=["cohort_week"])
    sessions["week_offset"] = ((
        sessions["session_week"].astype("datetime64[ns]")
        - sessions["cohort_week"].astype("datetime64[ns]")
    ) / np.timedelta64(1, "W")).astype(int)
    sessions = sessions[(sessions["week_offset"] >= 0) & (sessions["week_offset"] < lookback_weeks)]

    grouped = (
        sessions.groupby(["cohort_week", "week_offset"])
        ["user_id"].nunique().reset_index(name="active_users")
    )
    cohort_sizes = grouped[grouped["week_offset"] == 0].set_index("cohort_week")["active_users"]
    pivot = grouped.pivot(index="cohort_week", columns="week_offset", values="active_users").fillna(0)
    retention = pivot.div(cohort_sizes, axis=0).fillna(0)
    return retention


def latest_cohort_retention(retention: pd.DataFrame, weeks: int = 4) -> pd.Series:
    if retention.empty:
        return pd.Series(dtype=float)
    return retention.iloc[-1].iloc[:weeks]
