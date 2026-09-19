#!/bin/sh
set -e

# Run the pipeline (generate data, train models, score customers) whenever
# anything the app needs is missing. results/ ships with the project but
# models/ and data/processed/ do not, so checking results/ alone is not enough
# (the Live Triage page needs the trained models and the processed customers).
if [ ! -f results/scored_customers.csv ] \
  || [ ! -f models/fusion_model.joblib ] \
  || [ ! -d models/text_model ] \
  || [ ! -f data/processed/customers.jsonl ]; then
  echo "[entrypoint] Trained models or data not found yet - running the pipeline first."
  echo "[entrypoint] This trains the models and can take several minutes on first run."
  python src/ingest.py
  python src/preprocess.py
  python src/evaluate.py
fi

exec streamlit run app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
