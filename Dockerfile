FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch build (config.yaml pins device: cpu everywhere) - avoids
# pulling several GB of unused NVIDIA/CUDA packages that the default Linux
# PyPI wheel drags in.
RUN pip install --no-cache-dir --retries 20 --timeout 120 torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 20 --timeout 120 -r requirements.txt

COPY . .

# Strip Windows line endings (a zip made on Windows can carry them) or the
# script fails with a confusing "no such file or directory".
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
