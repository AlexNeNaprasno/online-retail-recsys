"""
На выходе:
  data/processed/sequences.parquet — одна строка на клиента:
    customer_id, item_sequence (список StockCode),
    seq_len, first_at, last_at
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.utils.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "processed" / "sequences.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_sequences() -> pd.DataFrame:
    engine = get_engine()
    print("Читаю marts.customer_sequences из PostgreSQL...")
    query = text("""
        SELECT customer_id, item_sequence, seq_len, first_at, last_at
        FROM marts.customer_sequences
        ORDER BY customer_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    print(f"  Клиентов: {len(df):,}")
    print(f"  Средняя длина: {df['seq_len'].mean():.1f}")
    return df


def main():
    df = load_sequences()
    df.to_parquet(OUT_PATH, index=False)
    print(f"Сохранил в {OUT_PATH}")


if __name__ == "__main__":
    main()