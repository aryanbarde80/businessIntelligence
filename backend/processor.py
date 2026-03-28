"""ETL helpers that clean simulated SaaS data."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

SIM_DIR = Path("data/simulated")
PROCESSED_DIR = Path("data/processed")


def _ensure_paths() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def read_simulated(name: str) -> pd.DataFrame:
    return pd.read_parquet(SIM_DIR / f"{name}.parquet")


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    users = users.drop_duplicates(subset=["user_id"]).copy()
    users["country"] = users["country"].fillna("Unknown")
    users["signup_date"] = pd.to_datetime(users["signup_date"])
    return users


def clean_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    sessions = sessions.drop_duplicates(subset=["session_id"]).dropna(subset=["user_id"]).copy()
    sessions["session_time"] = pd.to_datetime(sessions["session_time"])
    median_duration = sessions["duration"].dropna().median()
    sessions["duration"] = sessions["duration"].fillna(median_duration)
    return sessions


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.drop_duplicates(subset=["event_id"]).dropna(subset=["user_id"]).copy()
    events["event_time"] = pd.to_datetime(events["event_time"])
    events["feature"] = events["feature"].fillna("unknown")
    return events


def clean_payments(payments: pd.DataFrame) -> pd.DataFrame:
    payments = payments.copy()
    payments["payment_date"] = pd.to_datetime(payments["payment_date"])
    payments["plan"] = payments["plan"].str.lower()
    return payments


def aggregate_user_activity(users: pd.DataFrame, sessions: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    now = datetime.utcnow()
    session_metrics = (
        sessions.groupby("user_id").agg(
            sessions_total=("session_id", "count"),
            avg_session_duration=("duration", "mean"),
            first_session=("session_time", "min"),
            last_session=("session_time", "max"),
        )
    )
    session_metrics["avg_session_duration"] = session_metrics["avg_session_duration"].fillna(0)
    session_metrics["session_span_days"] = (
        (session_metrics["last_session"] - session_metrics["first_session"]).dt.days.fillna(0)
    )

    payment_metrics = (
        payments.groupby("user_id").agg(
            payments_total=("revenue", "sum"),
            last_payment=("payment_date", "max"),
            plans=("plan", lambda s: s.mode().iat[0] if not s.mode().empty else "free"),
            payment_count=("payment_date", "count"),
        )
    )
    payment_metrics["payments_total"] = payment_metrics["payments_total"].fillna(0)
    payment_metrics["last_payment"] = payment_metrics["last_payment"].fillna(pd.Timestamp("1970-01-01"))

    df = users.set_index("user_id").join(session_metrics, how="left").join(payment_metrics, how="left").fillna({
        "sessions_total": 0,
        "avg_session_duration": 0,
        "session_span_days": 0,
        "payments_total": 0,
        "payment_count": 0,
        "plans": "free",
    })
    df = df.reset_index()
    df["days_since_last_payment"] = (now - df["last_payment"]).dt.days.clip(lower=0)
    df["churn_flag"] = df["days_since_last_payment"] > 60
    df["churn_flag"] = df["churn_flag"] & (df["payment_count"] > 0)
    df["is_paid_user"] = df["plans"] != "free"
    df["avg_session_duration"] = df["avg_session_duration"].round(2)
    df["renewal_rate"] = df.apply(
        lambda row: min(1.0, row["payment_count"] / max(1, row["session_span_days"] + 1)), axis=1
    )
    return df


def export_processed(name: str, df: pd.DataFrame) -> None:
    _ensure_paths()
    df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)


def run_processing() -> Dict[str, pd.DataFrame]:
    _ensure_paths()
    users = clean_users(read_simulated("users"))
    sessions = clean_sessions(read_simulated("sessions"))
    events = clean_events(read_simulated("events"))
    payments = clean_payments(read_simulated("payments"))

    export_processed("users", users)
    export_processed("sessions", sessions)
    export_processed("events", events)
    export_processed("payments", payments)

    user_metrics = aggregate_user_activity(users, sessions, payments)
    export_processed("user_metrics", user_metrics)
    return {
        "users": users,
        "sessions": sessions,
        "events": events,
        "payments": payments,
        "user_metrics": user_metrics,
    }


if __name__ == "__main__":
    run_processing()
