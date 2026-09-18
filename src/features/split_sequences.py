"""
Каждого клиента превращаем в 3 сета:
  train = вся история, кроме последних 2 элементов
  val   = история, кроме последнего элемента (т.е. train + 1 элемент)
  test  = полная история

Также заменяем StockCode на индексы из vocab.json.

Сохраняем:
  data/processed/splits/train.parquet
  data/processed/splits/val.parquet
  data/processed/splits/test.parquet
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_PATH = ROOT / "data" / "processed" / "sequences.parquet"
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.json"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def load_vocab() -> dict:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Словарь: {len(vocab):,} записей (включая <UNK>=0)")
    return vocab


def encode_sequence(seq: list[str], vocab: dict) -> list[int]:
    """StockCode -> index. Неизвестные -> 0 (<UNK>)."""
    return [vocab.get(code, 0) for code in seq]


def main():
    print(f"Читаю {SEQUENCES_PATH}")
    df = pd.read_parquet(SEQUENCES_PATH)
    print(f"  Клиентов: {len(df):,}")

    vocab = load_vocab()

    # Минимальная длина, чтобы было что делить:
    # train >= 5, val >= 6, test >= 7
    MIN_LEN = 10
    df = df[df["seq_len"] >= MIN_LEN].reset_index(drop=True)
    print(f"  После фильтра seq_len >= {MIN_LEN}: {len(df):,} клиентов")

    train_rows, val_rows, test_rows = [], [], []

    for _, row in df.iterrows():
        seq = row["item_sequence"]
        encoded = encode_sequence(seq, vocab)

        train_seq = encoded[:-2]
        val_seq   = encoded[:-1]
        test_seq  = encoded

        base = {
            "customer_id": int(row["customer_id"]),
            "seq_len": int(row["seq_len"]),
        }
        train_rows.append({**base, "sequence": train_seq})
        val_rows.append({**base, "sequence": val_seq})
        test_rows.append({**base, "sequence": test_seq})

    train_df = pd.DataFrame(train_rows)
    val_df   = pd.DataFrame(val_rows)
    test_df  = pd.DataFrame(test_rows)

    train_path = SPLITS_DIR / "train.parquet"
    val_path   = SPLITS_DIR / "val.parquet"
    test_path  = SPLITS_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nTrain: {len(train_df):,} клиентов -> {train_path}")
    print(f"Val:   {len(val_df):,} клиентов -> {val_path}")
    print(f"Test:  {len(test_df):,} клиентов -> {test_path}")

    # Дополнительные проверки
    print("\nПроверки:")
    print(f"  Средняя длина train: {train_df['sequence'].apply(len).mean():.1f}")
    print(f"  Средняя длина val:   {val_df['sequence'].apply(len).mean():.1f}")
    print(f"  Средняя длина test:  {test_df['sequence'].apply(len).mean():.1f}")
    print(f"  Доля <UNK> в train:  {sum([s.count(0) for s in train_df['sequence']]) / max(1, sum(len(s) for s in train_df['sequence'])) * 100:.1f}%")


if __name__ == "__main__":
    main()