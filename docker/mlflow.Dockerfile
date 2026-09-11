FROM python:3.11-slim

# build-essential + libpq-dev are needed to compile psycopg2 from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        mlflow==2.16.2 \
        psycopg2==2.9.9 \
        boto3==1.34.162

EXPOSE 5000
