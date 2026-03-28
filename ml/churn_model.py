"""Churn prediction model training with pluggable backends."""
from __future__ import annotations

from pathlib import Path
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

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
    backend = os.getenv("MODEL_BACKEND", "logreg").lower()
    df = features.copy()
    X = df[FEATURE_COLUMNS]
    y = df["churn_flag"].astype(int)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception:  # pragma: no cover - optional dependency
            backend = "logreg"

    if backend == "xgboost":
        model = XGBClassifier(
            max_depth=4,
            n_estimators=160,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="auc",
            reg_lambda=1.0,
        )
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        base_importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        scaler = None
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(solver="liblinear", max_iter=500)
        model.fit(X_scaled, y)
        proba = model.predict_proba(X_scaled)[:, 1]
        base_importance = pd.Series(model.coef_[0], index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False)

    df["churn_probability"] = proba
    joblib.dump({"model": model, "scaler": scaler, "backend": backend}, MODEL_DIR / "churn_pipeline.joblib")

    report = classification_report(y, (proba >= 0.5).astype(int), output_dict=True)
    auc = roc_auc_score(y, proba)
    df.to_csv(MODEL_DIR / "churn_scored.csv", index=False)
    feature_importance = base_importance

    try:
        pi = permutation_importance(model, X if backend == "xgboost" else X_scaled, y, n_repeats=10, random_state=42)
        perm_importance = pd.Series(pi.importances_mean, index=FEATURE_COLUMNS).sort_values(ascending=False)
        feature_importance = pd.concat(
            [feature_importance.rename("model_importance"), perm_importance.rename("permutation_importance")],
            axis=1,
        )
    except Exception:
        feature_importance = feature_importance.rename("model_importance").to_frame()

    feature_importance.to_csv(MODEL_DIR / "churn_feature_importance.csv")

    return {
        "model": model,
        "scaler": scaler,
        "metrics": {"auc": auc, "report": report},
        "backend": backend,
        "feature_importance": feature_importance.to_dict(),
        "scored": df,
    }


def load_churn_pipeline() -> dict:
    artifact = joblib.load(MODEL_DIR / "churn_pipeline.joblib")
    return artifact
