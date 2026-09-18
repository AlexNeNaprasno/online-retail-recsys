"""
Идея:
- Считаем частоту каждого StockCode по всем последовательностям.
- Берём топ-N (N=3000) самых популярных.
- Остальные кодируем как <UNK>.
- Индекс 0 зарезервирован под <UNK>, индексы 1..N — под реальные товары.

На выходе:
  data/processed/vocab.json — {stock_code: index}
  data/processed/vocab_meta.json — {n_items, n_total, n_unknown}
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES_PATH = ROOT / "data" / "processed" / "sequences.parquet"
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.json"
META_PATH = ROOT / "data" / "processed" / "vocab_meta.json"

# Сколько товаров оставить в словаре
TOP_N = 3000

# Специальные токены
UNK_TOKEN = "<UNK>"   # неизвестный товар
PAD_TOKEN = "<PAD>"   # padding (добавим позже, если понадобится)


def build_vocab(sequences: list[list[str]], top_n: int) -> dict:
    """Считает частоты и возвращает словарь {stock_code: index}."""
    counter = Counter()
    for seq in sequences:
        counter.update(seq)

    # Топ-N по частоте
    top_items = [item for item, _ in counter.most_common(top_n)]

    # <UNK> получает индекс 0, реальные товары — с 1
    vocab = {UNK_TOKEN: 0}
    for i, item in enumerate(top_items, start=1):
        vocab[item] = i

    return vocab, counter


def main():
    print(f"Читаю {SEQUENCES_PATH}")
    df = pd.read_parquet(SEQUENCES_PATH)
    sequences = df["item_sequence"].tolist()
    print(f"  Клиентов: {len(sequences):,}")

    print(f"Строю словарь (топ-{TOP_N})...")
    vocab, counter = build_vocab(sequences, TOP_N)

    n_total = len(counter)
    n_kept = len(vocab) - 1  # минус UNK
    n_unk = n_total - n_kept

    print(f"  Всего уникальных StockCode: {n_total:,}")
    print(f"  Оставлено в словаре:        {n_kept:,}")
    print(f"  Пойдут в <UNK>:             {n_unk:,} ({n_unk/n_total*100:.1f}%)")

    # Сохраняем
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "n_items": n_kept,
            "n_total": n_total,
            "n_unknown": n_unk,
            "top_n": TOP_N,
            "unk_index": 0,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nСохранил словарь: {VOCAB_PATH}")
    print(f"Сохранил мета:    {META_PATH}")


if __name__ == "__main__":
    main()