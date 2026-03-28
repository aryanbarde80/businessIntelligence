"""Ingest cleaned data into PostgreSQL (best-effort)."""
from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from backend.db import get_engine

PROCESSED_DIR = Path("data/processed")
TABLES = ["users", "sessions", "events", "payments", "user_metrics"]


def read_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", parse_dates=True)


def ingest() -> None:
    """Load processed CSVs into the configured database.

    If the database is unavailable (e.g., Postgres container not running), we
    log and continue so the rest of the pipeline can still produce artifacts
    and dashboard data.
    """

    try:
        engine = get_engine()
        with engine.begin() as connection:
            for table in TABLES:
                df = read_processed(table)
                df.to_sql(table, connection, if_exists="replace", index=False)
        logging.info("Ingestion completed successfully")
    except SQLAlchemyError as exc:
        logging.warning("Skipping DB ingest; database not reachable (%s)", exc)


def run_ingestion() -> None:
    ingest()


if __name__ == "__main__":
    run_ingestion()
