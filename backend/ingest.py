"""Ingest cleaned data into PostgreSQL."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.db import get_engine

PROCESSED_DIR = Path("data/processed")
TABLES = ["users", "sessions", "events", "payments", "user_metrics"]


def read_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", parse_dates=True)


def ingest() -> None:
    engine = get_engine()
    with engine.begin() as connection:
        for table in TABLES:
            df = read_processed(table)
            df.to_sql(table, connection, if_exists="replace", index=False)


def run_ingestion() -> None:
    ingest()


if __name__ == "__main__":
    run_ingestion()
