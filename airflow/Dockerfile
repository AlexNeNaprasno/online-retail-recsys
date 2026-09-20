FROM apache/airflow:2.8.0-python3.11

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY airflow/requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir --retries 5 -r /requirements-airflow.txt