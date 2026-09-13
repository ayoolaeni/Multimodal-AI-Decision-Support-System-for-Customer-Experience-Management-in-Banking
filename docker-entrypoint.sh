#!/bin/sh
set -e

if [ ! -f results/scored_customers.csv ]; then
  echo "[entrypoint] No scored results found yet - running the pipeline first."
  echo "[entrypoint] This trains the models and can take several minutes on first run."
  python src/ingest.py
  python src/preprocess.py
  python src/evaluate.py
fi

exec streamlit run app/streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true
