"""Churn prediction model training."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("artifacts")
FEATURE_COLUMNS = [
    "sessions_total",
    "avg_session_duration",
    "session_span_days",
    "payments_total",
    "payment_count",
    "renewal_rate",
]


def train_churn_model(features: pd.DataFrame) -> dict:
    df = features.copy()
    X = df[FEATURE_COLUMNS]
    y = df["churn_flag"].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(solver="liblinear", max_iter=500)
    model.fit(X_scaled, y)

    df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_DIR / "churn_pipeline.joblib")

    report = classification_report(y, model.predict(X_scaled), output_dict=True)
    auc = roc_auc_score(y, df["churn_probability"])
    df.to_csv(MODEL_DIR / "churn_scored.csv", index=False)
    return {"model": model, "scaler": scaler, "metrics": {"auc": auc, "report": report}, "scored": df}


def load_churn_pipeline() -> dict:
    artifact = joblib.load(MODEL_DIR / "churn_pipeline.joblib")
    return artifact
