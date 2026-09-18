"""
Загружает очищенный Parquet в PostgreSQL.

Создаёт:
  staging.retail_clean — очищенные транзакции

Логирует запуск в meta.etl_log.
"""
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ------- Пути -------
ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "data" / "processed" / "retail_clean.parquet"

# ------- Переменные окружения -------
load_dotenv(ROOT / ".env")


def get_engine():
    """Собираем строку подключения и создаём SQLAlchemy engine."""
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def log_start(engine, step: str) -> int:
    """Записываем начало шага в meta.etl_log. Возвращаем id."""
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO meta.etl_log (step, status)
                VALUES (:step, 'started')
                RETURNING id
            """),
            {"step": step},
        )
        return result.scalar_one()


def log_end(engine, log_id: int, rows: int, status: str, comment: str = ""):
    """Обновляем запись в meta.etl_log по завершении шага."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE meta.etl_log
                   SET finished_at = NOW(),
                       rows_loaded = :rows,
                       status      = :status,
                       comment     = :comment
                 WHERE id = :id
            """),
            {"id": log_id, "rows": rows, "status": status, "comment": comment},
        )


def load_parquet() -> pd.DataFrame:
    """Читаем очищенный parquet."""
    print(f"Читаю {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Строк: {len(df):,}")
    return df


def upload_to_postgres(df: pd.DataFrame, engine) -> int:
    """Заливаем в staging.retail_clean."""
    print("Заливаю в staging.retail_clean...")
    df.to_sql(
        name="retail_clean",
        con=engine,
        schema="staging",
        if_exists="replace",   # для простоты перезаписываем
        index=False,
        chunksize=10_000,
        method="multi",
    )
    return len(df)


def main():
    engine = get_engine()
    step = "load_staging_retail_clean"
    log_id = log_start(engine, step)
    t0 = time.time()

    try:
        df = load_parquet()
        rows = upload_to_postgres(df, engine)
        elapsed = time.time() - t0
        log_end(engine, log_id, rows, "success", f"elapsed={elapsed:.1f}s")
        print(f"Готово. Загружено {rows:,} строк за {elapsed:.1f} сек.")
    except Exception as e:
        log_end(engine, log_id, 0, "failed", str(e))
        print(f"Ошибка: {e}")
        raise


if __name__ == "__main__":
    main()