"""Synthetic data simulation for SaaS analytics."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

faker = Faker()
COUNTRIES = ["United States", "India", "Canada", "United Kingdom", "Germany", "Australia", "Brazil", "France", "Japan"]
EVENT_TYPES = ["click", "view", "feature_use", "submit", "toggle", "navigate"]
FEATURES = ["billing", "reports", "dashboards", "collaboration", "alerts", "integrations"]
PLANS = [
    {"plan": "free", "price": 0},
    {"plan": "starter", "price": 49},
    {"plan": "growth", "price": 129},
    {"plan": "enterprise", "price": 499},
]
OUTPUT_DIR = Path("data/simulated")


def make_date(start: datetime, end: datetime) -> datetime:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(1, delta)))


def generate_users(num_users: int = 1200, days: int = 540) -> pd.DataFrame:
    now = datetime.utcnow()
    start_window = now - timedelta(days=days)
    records = []
    for user_id in range(1, num_users + 1):
        signup = make_date(start_window, now)
        records.append(
            {
                "user_id": user_id,
                "signup_date": signup,
                "country": random.choice(COUNTRIES),
            }
        )
    df = pd.DataFrame(records)
    df = df.sort_values("signup_date").reset_index(drop=True)

    dup = df.sample(max(1, int(0.01 * len(df))), random_state=42)
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(frac=0.02, random_state=1).index, "country"] = None
    return df


def generate_sessions(users: pd.DataFrame) -> pd.DataFrame:
    now = datetime.utcnow()
    records = []
    session_id = 1
    for _index, user in users.iterrows():
        count = random.randint(5, 25)
        for _ in range(count):
            session_time = make_date(user["signup_date"], now)
            duration = random.gauss(32, 10)
            if duration < 2:
                duration = abs(duration) + 2
            if random.random() < 0.05:
                duration = None
            records.append(
                {
                    "session_id": session_id,
                    "user_id": user["user_id"],
                    "session_time": session_time,
                    "duration": round(duration, 2) if duration else None,
                }
            )
            session_id += 1
    df = pd.DataFrame(records)
    df.loc[df.sample(frac=0.01, random_state=2).index, "session_time"] = None
    return df


def generate_events(sessions: pd.DataFrame) -> pd.DataFrame:
    records = []
    event_id = 1
    for _, session in sessions.iterrows():
        count = random.randint(3, 8)
        base_time = session["session_time"]
        for offset in range(count):
            timestamp = base_time + timedelta(seconds=random.randint(10, 300)) if pd.notnull(base_time) else datetime.utcnow()
            event_type = random.choice(EVENT_TYPES)
            feature = random.choice(FEATURES)
            records.append(
                {
                    "event_id": event_id,
                    "user_id": session["user_id"],
                    "session_id": session["session_id"],
                    "event_type": event_type,
                    "feature": feature,
                    "event_time": timestamp,
                }
            )
            event_id += 1
    df = pd.DataFrame(records)
    df.loc[df.sample(frac=0.01, random_state=3).index, "feature"] = None
    return df


def generate_payments(users: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, user in users.drop_duplicates(subset=["user_id"]).iterrows():
        plan = random.choices(PLANS, weights=[0.35, 0.25, 0.25, 0.15])[0]
        months = random.randint(0, 12) if plan["price"] > 0 else 0
        base_date = user["signup_date"] + timedelta(days=15)
        if months == 0 and plan["price"] > 0:
            months = 1
        for m in range(months):
            pay_date = base_date + timedelta(days=30 * m) + timedelta(days=random.randint(0, 4))
            records.append(
                {
                    "user_id": user["user_id"],
                    "plan": plan["plan"],
                    "revenue": plan["price"],
                    "payment_date": pay_date,
                }
            )
        if plan["price"] == 0:
            records.append(
                {
                    "user_id": user["user_id"],
                    "plan": plan["plan"],
                    "revenue": 0,
                    "payment_date": base_date,
                }
            )
    df = pd.DataFrame(records)
    mask_idx = df.sample(frac=0.01, random_state=4).index
    df.loc[mask_idx, "plan"] = df.loc[mask_idx, "plan"].str.upper()
    return df


def persist_tables(users: pd.DataFrame, sessions: pd.DataFrame, events: pd.DataFrame, payments: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    users.to_parquet(OUTPUT_DIR / "users.parquet", index=False)
    sessions.to_parquet(OUTPUT_DIR / "sessions.parquet", index=False)
    events.to_parquet(OUTPUT_DIR / "events.parquet", index=False)
    payments.to_parquet(OUTPUT_DIR / "payments.parquet", index=False)
    users.to_csv(OUTPUT_DIR / "users.csv", index=False)
    sessions.to_csv(OUTPUT_DIR / "sessions.csv", index=False)
    events.to_csv(OUTPUT_DIR / "events.csv", index=False)
    payments.to_csv(OUTPUT_DIR / "payments.csv", index=False)


def run_simulation(num_users: int = 1200) -> dict:
    users = generate_users(num_users=num_users)
    sessions = generate_sessions(users)
    events = generate_events(sessions)
    payments = generate_payments(users)
    persist_tables(users, sessions, events, payments)
    return {
        "users": users,
        "sessions": sessions,
        "events": events,
        "payments": payments,
    }


if __name__ == "__main__":
    run_simulation()
