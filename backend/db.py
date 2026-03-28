"""PostgreSQL connection helpers."""
from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

DEFAULT_URL = "postgresql+psycopg2://analytics_user:secretpass@localhost:5432/analytics"


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_URL)


def get_engine() -> Engine:
    url = get_database_url()
    return create_engine(url, future=True)


def get_connection() -> Engine:
    return get_engine().connect()
