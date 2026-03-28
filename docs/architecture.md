# Platform architecture

- `data/generator.py`: synthetic user/session/event/payment tables.
- `backend/processor.py`: cleans simulated data and exports `data/processed/*.csv`.
- `backend/ingest.py`: writes tables into Postgres using SQLAlchemy.
- `analytics/`: cohort, conversion, segmentation, and churn helpers consume processed tables.
- `ml/`: feature assembler + logistic regression churn model saved under `artifacts/`.
- `insights/`: rule-based insight generator that highlights high-churn countries, valuable features, and best plans.
- `dashboard/app.py`: Streamlit interface with KPIs, funnel, cohort heatmap, daily DAU trend, segmentation table, and insight bullets drawn directly from the processed data.

Artifacts:
- `artifacts/analytics_summary.json`: JSON summary of conversion, churn, and top countries.
- `artifacts/cohort_retention.csv`: retention matrix for latest cohorts.
- `artifacts/churn_pipeline.joblib`: serialized scaler + model.
- `artifacts/churn_scored.csv`: scored dataset with churn probability per user.
- `artifacts/insights.md`: (generated via `/run_pipeline` when insights change)
