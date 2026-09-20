"""
DAG: retail_pipeline

Оркеструет DE-часть проекта:
  raw CSV → очистка → валидация → PostgreSQL → витрины → последовательности.

ML-обучение вынесено за скобки (отдельный пайплайн).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text

# ==== Пути внутри контейнера ====
PROJECT_ROOT = Path("/opt/airflow/project")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED / "splits"

RAW_CSV = DATA_RAW / "online_retail.csv"
CLEAN_PARQUET = DATA_PROCESSED / "retail_clean.parquet"
VOCAB_JSON = DATA_PROCESSED / "vocab.json"

# ==== Postgres ====
PG_HOST = os.environ.get("PG_HOST", "postgres")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.environ.get("PG_USER", "retail")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "retail")
PG_DB = os.environ.get("PG_DB", "retail")

PG_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

SERVICE_CODES = {
    "POST", "DOT", "M", "BANK CHARGES", "AMAZONFEE",
    "C2", "D", "S", "TEST", "ADJUST", "B", "CRUK",
}


def task_check_raw():
    """Проверяем, что сырой CSV на месте."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Нет сырого файла: {RAW_CSV}")
    size_mb = RAW_CSV.stat().st_size / 1024 / 1024
    print(f"OK: {RAW_CSV} ({size_mb:.1f} MB)")


def task_clean_data():
    """Очистка Online Retail: отмены, возвраты, служебные SKU, гости."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_CSV, encoding="ISO-8859-1")
    n0 = len(df)
    print(f"Сырых строк: {n0:,}")

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[df["CustomerID"].notna()]
    df = df[~df["StockCode"].astype(str).str.upper().isin(SERVICE_CODES)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="ISO8601")
    df["CustomerID"] = df["CustomerID"].astype(int)

    desc_map = (
        df.dropna(subset=["Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    df["Description"] = df["Description"].fillna(df["StockCode"].map(desc_map))
    df = df.dropna(subset=["Description"])
    df = df.drop_duplicates()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df = df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"Очищено: {len(df):,} строк ({n0 - len(df):,} удалено)")
    print(f"Клиентов: {df['CustomerID'].nunique():,}, товаров: {df['StockCode'].nunique():,}")


def task_validate_clean():
    """Проверки качества после очистки."""
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"Нет очищенного parquet: {CLEAN_PARQUET}")

    df = pd.read_parquet(CLEAN_PARQUET)

    assert len(df) > 300_000, f"Мало строк после очистки: {len(df):,}"
    assert df["CustomerID"].notna().all(), "Есть null CustomerID"
    assert (df["Quantity"] > 0).all(), "Есть Quantity <= 0"
    assert (df["UnitPrice"] > 0).all(), "Есть UnitPrice <= 0"
    assert df["InvoiceDate"].dtype.kind == "M", "InvoiceDate не datetime"

    print(f"OK: {len(df):,} строк, {df['CustomerID'].nunique():,} клиентов")


def task_load_to_postgres():
    """Загружаем очищенные данные в Postgres (raw_retail)."""
    df = pd.read_parquet(CLEAN_PARQUET)
    df.columns = [c.lower() for c in df.columns]

    engine = create_engine(PG_URL)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS raw_retail"))

    df.to_sql("raw_retail", engine, if_exists="replace", index=False, chunksize=10_000)
    print(f"Загружено {len(df):,} строк в raw_retail")


def task_build_marts():
    """Строим dim_customer, dim_product, fact_sales."""
    engine = create_engine(PG_URL)

    sql_marts = """
    DROP TABLE IF EXISTS dim_customer CASCADE;
    CREATE TABLE dim_customer AS
    SELECT
        customerid                                   AS customer_id,
        MIN(invoicedate)                             AS first_order,
        MAX(invoicedate)                             AS last_order,
        COUNT(DISTINCT invoiceno)                    AS n_orders,
        SUM(totalprice)                              AS total_revenue,
        COUNT(DISTINCT stockcode)                    AS n_distinct_items,
        MAX(country)                                 AS country
    FROM raw_retail
    GROUP BY customerid;

    DROP TABLE IF EXISTS dim_product CASCADE;
    CREATE TABLE dim_product AS
    SELECT
        stockcode                                    AS stock_code,
        MAX(description)                             AS description,
        AVG(unitprice)                               AS avg_price,
        SUM(quantity)                                AS total_quantity,
        SUM(totalprice)                              AS total_revenue,
        COUNT(DISTINCT customerid)                   AS n_customers
    FROM raw_retail
    GROUP BY stockcode;

    DROP TABLE IF EXISTS fact_sales CASCADE;
    CREATE TABLE fact_sales AS
    SELECT
        invoiceno                                    AS invoice_no,
        stockcode                                    AS stock_code,
        customerid                                   AS customer_id,
        invoicedate                                  AS invoice_date,
        quantity,
        unitprice,
        totalprice,
        country
    FROM raw_retail;

    CREATE INDEX IF NOT EXISTS idx_fact_sales_customer ON fact_sales(customer_id);
    CREATE INDEX IF NOT EXISTS idx_fact_sales_product  ON fact_sales(stock_code);
    CREATE INDEX IF NOT EXISTS idx_fact_sales_date     ON fact_sales(invoice_date);
    """
    with engine.begin() as conn:
        for stmt in sql_marts.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))

    with engine.begin() as conn:
        for tbl in ("dim_customer", "dim_product", "fact_sales"):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            print(f"{tbl}: {n:,} строк")


def task_build_sequences():
    """Последовательности покупок на клиента → parquet."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CLEAN_PARQUET)
    df = df.sort_values(["CustomerID", "InvoiceDate"])

    # Агрегируем один товар в один счёт (для чистоты последовательности)
    grouped = (
        df.groupby(["CustomerID", "InvoiceNo"], as_index=False)
          .agg({"StockCode": "first", "InvoiceDate": "min"})
          .sort_values(["CustomerID", "InvoiceDate"])
    )

    seqs = grouped.groupby("CustomerID")["StockCode"].apply(list)
    seqs = seqs[seqs.apply(len) >= 10]
    print(f"Клиентов с >= 10 покупками: {len(seqs):,}")

    rows = []
    for cid, seq in seqs.items():
        rows.append({
            "customer_id": cid,
            "sequence": seq,
            "len": len(seq),
        })

    out = SPLITS_DIR / "customer_sequences.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Сохранено: {out}")


def task_notify_done():
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


default_args = {
    "owner": "online-retail-recsys",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="retail_pipeline",
    description="Online Retail DE pipeline: clean → validate → Postgres → marts → sequences",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # ручной запуск
    catchup=False,
    tags=["retail", "de", "etl"],
) as dag:

    t1 = PythonOperator(task_id="check_raw_data", python_callable=task_check_raw)
    t2 = PythonOperator(task_id="clean_data", python_callable=task_clean_data)
    t3 = PythonOperator(task_id="validate_clean", python_callable=task_validate_clean)
    t4 = PythonOperator(task_id="load_to_postgres", python_callable=task_load_to_postgres)
    t5 = PythonOperator(task_id="build_marts", python_callable=task_build_marts)
    t6 = PythonOperator(task_id="build_sequences", python_callable=task_build_sequences)
    t7 = BashOperator(task_id="notify_done", bash_command='echo "DAG finished"')

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7