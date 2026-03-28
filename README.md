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

### Data guarantees
- The dashboard auto-detects when `data/processed/*.csv` are missing and runs `python run_pipeline.py` (which in turn runs the generator + processor) before rendering; this ensures `user_metrics.csv` always exists even on a fresh clone or Render deployment.
- Artifacts (`artifacts/churn_scored.csv`, `analytics_summary.json`, `insights.md`, etc.) refresh each time the orchestration script completes, so the dashboard sees the freshest scores and insights.

## Prerequisites
- Python 3.10+ with `pip`
- `docker` & `docker-compose` (for PostgreSQL)
- Optional: `streamlit` to view the dashboard
- Optional: Render/Streamlit Cloud account for deployment

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
   - If Postgres isn't running, the pipeline will still finish and emit CSVs/artifacts; it will simply log that DB ingest was skipped.
4. Launch the Streamlit dashboard:
   ```sh
   streamlit run dashboard/app.py --server.port 8501
   ```
5. Optionally inspect SQL queries at `backend/sql/queries.sql` for DAU/MAU, retention, and funnel analysis.

## Render deployment checklist
1. Push your repo to GitHub, add a new **Web Service** in Render, and point it to `main`.
2. Set the commands via Render’s UI:
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `streamlit run dashboard/app.py --server.port $PORT --server.headless true`
3. Add environment variables (e.g., `DATABASE_URL`) in the Render dashboard if you use a managed Postgres instance.
4. Render exposes `$PORT`, `RENDER_EXTERNAL_URL`, and other metadata; the dashboard already respects `$PORT` so no extra code change is required.
5. Paid plans keep the service always-on; free tiers may spin down after inactivity—schedule a health-check or manual trigger for consistent availability.

## Troubleshooting
- `FileNotFoundError` for `data/processed/*.csv`: manually run `python run_pipeline.py` or delete the CSVs and restart the dashboard so `_ensure_processed_tables` regenerates them.
- Database not reachable: ensure `docker-compose up -d` is running; if not, the pipeline now proceeds and skips ingestion while still writing CSVs/artifacts.
- Watch the console/logs for `[INFO] Generating synthetic data`, `Training churn model`, or `Submitting CSVs` to confirm each pipeline stage completed.
- To refresh insights, delete `artifacts/*.json`/`.csv` (or rerun the pipeline) so the dashboard reruns `rules.generate_insights`.

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
- `artifacts/dau_anomalies.csv`: Daily active users with z-scores and anomaly flags.
- `artifacts/analytics_summary.json` now also includes LTV and anomaly snippets for downstream consumers.
