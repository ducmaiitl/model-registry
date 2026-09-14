FROM python:3.11-slim

# curl backs the compose healthcheck; build-essential + libpq-dev compile psycopg2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# google-cloud-storage is what lets MLflow read/write gs:// artifact URIs.
RUN pip install --no-cache-dir \
        mlflow==2.16.2 \
        psycopg2==2.9.9 \
        google-cloud-storage==2.18.2

EXPOSE 5000
