"""Orchestrates the full analytics stack end-to-end."""
from __future__ import annotations

import json
from pathlib import Path
import logging

from analytics import cohorts, conversion, segmentation, churn as churn_analysis
from analytics.anomaly import detect_dau_anomalies
from analytics.ltv import ltv_summary
from backend.ingest import ingest
from backend.processor import run_processing
from data.generator import run_simulation
from insights.rules import generate_insights
from ml.churn_model import train_churn_model
from ml.features import prepare_ml_features

ARTIFACT_DIR = Path("artifacts")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def run_pipeline() -> None:
    ARTIFACT_DIR.mkdir(exist_ok=True)

    logging.info("1. Generating synthetic data")
    run_simulation()

    logging.info("2. Cleaning data")
    processed = run_processing()

    logging.info("3. Loading into PostgreSQL")
    ingest()

    logging.info("4. Computing analytics summaries")
    retention = cohorts.build_weekly_cohort(processed["sessions"])
    retention.to_csv(ARTIFACT_DIR / "cohort_retention.csv", index=True)

    anomalies = detect_dau_anomalies(processed["sessions"])
    anomalies.to_csv(ARTIFACT_DIR / "dau_anomalies.csv", index=False)

    ltv = ltv_summary(processed["user_metrics"])

    insight_texts = generate_insights(processed["user_metrics"], processed["events"])
    summary = {
        "conversion": conversion.conversion_rates(processed["user_metrics"]).to_dict(),
        "churn": churn_analysis.churn_summary(processed["user_metrics"]).to_dict(),
        "top_countries": segmentation.segment_by_country(processed["user_metrics"]).head(3)["churn_rate"].to_dict(),
        "ltv": ltv.to_dict(),
        "dau_anomalies": anomalies[anomalies["is_anomaly"]].to_dict(orient="list"),
        "churn_backend": model_artifacts.get("backend", "logreg"),
        "feature_importance": (
            model_artifacts.get("feature_importance", {})
            if isinstance(model_artifacts, dict)
            else {}
        ),
        "insights": insight_texts,
    }

    with (ARTIFACT_DIR / "analytics_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    with (ARTIFACT_DIR / "insights.md").open("w", encoding="utf-8") as fh:
        if insight_texts:
            for insight in insight_texts:
                fh.write(f"- {insight}\n")
        else:
            fh.write("Insights will populate after pipeline runs with richer data.\n")

    logging.info("5. Training churn model")
    features = prepare_ml_features(processed["user_metrics"])
    model_artifacts = train_churn_model(features)

    logging.info("6. Pipeline complete")
    logging.info("Model AUC: %.3f", model_artifacts["metrics"]["auc"])


if __name__ == "__main__":
    run_pipeline()
