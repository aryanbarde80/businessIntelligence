"""Forecasting utilities for activity metrics."""
from __future__ import annotations

import pandas as pd


def forecast_dau(sessions: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
    """Return a forecast for DAU using simple exponential smoothing.

    Falls back gracefully if statsmodels is unavailable or data is insufficient.
    """
    if sessions.empty or "session_time" not in sessions:
        return pd.DataFrame(columns=["date", "dau", "forecast"])

    daily = (
        sessions.dropna(subset=["session_time"])
        .groupby(sessions["session_time"].dt.date)["user_id"]
        .nunique()
        .reset_index(name="dau")
        .sort_values("session_time", ascending=True)
    )
    daily = daily.rename(columns={"session_time": "date"})

    if len(daily) < 7:
        return pd.DataFrame(columns=["date", "dau", "forecast"])

    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    except Exception:
        return pd.DataFrame(columns=["date", "dau", "forecast"])

    model = SimpleExpSmoothing(daily["dau"], initialization_method="estimated")
    fit = model.fit()
    forecast_index = pd.date_range(start=daily["date"].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
    forecast_values = fit.forecast(periods)

    forecast_df = pd.DataFrame({"date": forecast_index.date, "dau": None, "forecast": forecast_values})
    hist_df = daily[["date", "dau"]].copy()
    hist_df["forecast"] = None
    combined = pd.concat([hist_df, forecast_df], ignore_index=True)
    return combined
