# AI-Powered SaaS Analytics Platform

This project simulates, ingests, analyzes, and visualizes SaaS user data end-to-end. It produces clean datasets, stores them in PostgreSQL, derives key analytics, trains a churn-prediction model, generates insights, and exposes everything through a Streamlit dashboard.

## Architecture
1. **Data Simulation** (`data/generator.py`): creates synthetic users, sessions, events, and payments with realistic patterns.
2. **Processing & Ingestion** (`backend/processor.py`, `backend/ingest.py`): cleans datasets and writes them to PostgreSQL.
3. **Analytics** (`analytics/`): computes cohorts, segment stats, conversion/funnel metrics, and churn summaries.
4. **Machine Learning** (`ml/churn_model.py`): trains a logistic-regression churn model and saves probabilities.
5. **Insights** (`insights/rules.py`): derives rule-based observations for drop-offs and retention.
6. **Dashboard** (`dashboard/app.py`): Streamlit UI that visualizes KPIs, funnels, cohorts, churn probabilities, and insights.
7. **Deployment**: Can be run locally with Dockerized Postgres and the Streamlit app.

## Prerequisites
- Python 3.10+ with `pip`
- `docker` & `docker-compose` (for PostgreSQL)
- Optional: `streamlit` to view the dashboard

## Setup & Run
1. Install Python requirements:
   ```sh
   pip install -r requirements.txt
   ```
2. Start PostgreSQL:
   ```sh
   docker-compose up -d
   ```
   - Database: `analytics`
   - User/password: `analytics_user` / `secretpass`
3. Run the orchestration script:
   ```sh
   python run_pipeline.py
   ```
   This generates synthetic data, cleans it, ingests it into Postgres, runs analytics, trains the churn model, and refreshes the CSV exports and insights used by the dashboard.
4. Launch the Streamlit dashboard:
   ```sh
   streamlit run dashboard/app.py --server.port 8501
   ```
5. Optionally inspect SQL queries at `backend/sql/queries.sql` for DAU/MAU, retention, and funnel analysis.

## Folder Highlights
- `data/`: synthetic data generator plus exported CSV snapshots.
- `backend/`: ETL helpers and SQL query library.
- `analytics/`: cohort, segmentation, and churn analysis helpers.
- `ml/`: churn-feature engineering and model training.
- `insights/`: simple generator that translates trends into sentences.
- `dashboard/`: Streamlit UI that reads processed CSVs and highlights KPIs.
- `run_pipeline.py`: master script that glues everything together.

## Next Steps
1. Point the dashboard at a hosted PostgreSQL/Postgres-compatible data warehouse.
2. Swap the logistic regressor for XGBoost by setting the `MODEL_BACKEND` flag.
3. Deploy the dashboard via Streamlit Cloud, Vercel, or a container with `streamlit run`.

## Artifacts & Outputs
- `artifacts/analytics_summary.json`: JSON summary of conversion, churn, and top countries used for reporting.
- `artifacts/cohort_retention.csv`: Latest weekly retention matrix to instrument cohort reviews.
- `artifacts/insights.md`: Rule-based observations surfaced to the dashboard.
- `artifacts/churn_pipeline.joblib`: Serialized scaler and logistic-regression model pipeline.
- `artifacts/churn_scored.csv`: Scored user metrics with churn probabilities for downstream alerts.
