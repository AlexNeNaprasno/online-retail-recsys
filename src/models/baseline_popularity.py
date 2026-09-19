"""
1. Считаем частоту каждого товара в train-сете.
2. Топ-N самых частых — это наша "модель".
3. Оцениваем на val и test:
   - Для каждого клиента правильный ответ = последний элемент его последовательности.
   - Смотрим, попал ли ответ в топ-N популярных.

Метрики: HitRate@K, NDCG@K, MRR@K.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")
    val   = pd.read_parquet(SPLITS_DIR / "val.parquet")
    test  = pd.read_parquet(SPLITS_DIR / "test.parquet")
    return train, val, test


def count_popularity(train_df: pd.DataFrame) -> Counter:
    """Считаем частоту каждого товара в train-сете."""
    counter = Counter()
    for seq in train_df["sequence"]:
        counter.update(seq)
    # Убираем <UNK> (индекс 0) — не рекомендуем неизвестные товары
    counter.pop(0, None)
    return counter


def top_k_items(counter: Counter, k: int) -> list[int]:
    """Возвращает топ-K самых частых товаров (индексы)."""
    return [item for item, _ in counter.most_common(k)]


def hit_rate_at_k(preds: list[int], target: int, k: int) -> float:
    """Попал ли target в первые k из preds."""
    return 1.0 if target in preds[:k] else 0.0


def ndcg_at_k(preds: list[int], target: int, k: int) -> float:
    """NDCG@k для одного пользователя."""
    for i, p in enumerate(preds[:k], start=1):
        if p == target:
            return 1.0 / np.log2(i + 1)
    return 0.0


def mrr_at_k(preds: list[int], target: int, k: int) -> float:
    """MRR@k для одного пользователя."""
    for i, p in enumerate(preds[:k], start=1):
        if p == target:
            return 1.0 / i
    return 0.0


def evaluate(split_df: pd.DataFrame, top_items: list[int], k: int = 10) -> dict:
    """Считаем средние метрики по всем клиентам."""
    hits, ndcgs, mrrs = [], [], []
    for seq in split_df["sequence"]:
        if len(seq) < 2:
            continue
        target = seq[-1]           # правильный следующий товар
        # предсказание — статичный топ-K популярных
        hits.append(hit_rate_at_k(top_items, target, k))
        ndcgs.append(ndcg_at_k(top_items, target, k))
        mrrs.append(mrr_at_k(top_items, target, k))

    return {
        "n_users": len(hits),
        f"HitRate@{k}": float(np.mean(hits)),
        f"NDCG@{k}":    float(np.mean(ndcgs)),
        f"MRR@{k}":     float(np.mean(mrrs)),
    }


def main():
    train, val, test = load_splits()
    print(f"Train: {len(train):,} клиентов")
    print(f"Val:   {len(val):,} клиентов")
    print(f"Test:  {len(test):,} клиентов")

    counter = count_popularity(train)
    print(f"\nУникальных товаров в train: {len(counter):,}")

    K = 10
    top_items = top_k_items(counter, K)

    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    print(f"\nТоп-{K} популярных товаров (StockCode):")
    for rank, item in enumerate(top_items, 1):
        code = inv_vocab.get(item, "?")
        count = counter[item]
        print(f"  {rank:>2}. {code} (встречается {count:,} раз)")

    # Оценка
    val_metrics = evaluate(val, top_items, K)
    test_metrics = evaluate(test, top_items, K)

    print(f"\n=== VAL ===")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n=== TEST ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    # Сохраняем
    results = {
        "baseline": "popularity",
        "K": K,
        "val": val_metrics,
        "test": test_metrics,
        "top_items": [int(x) for x in top_items],
    }
    out_path = RESULTS_DIR / "baseline_popularity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nСохранил метрики: {out_path}")


if __name__ == "__main__":
    main()