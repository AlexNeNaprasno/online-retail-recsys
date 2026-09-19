"""
PyTorch Dataset для GRU4Rec.

Идея: для каждой последовательности режем её на окна длины WINDOW.
  input: [i_t-WINDOW+1, ..., i_t]
  target: i_{t+1}

Если последовательность короче окна — паддинг нулями слева.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

WINDOW = 20  
PAD_IDX = 0  


class SequenceDataset(Dataset):


    def __init__(self, sequences: list[list[int]], window: int = WINDOW):
        self.window = window
        self.samples = []  

        for seq in sequences:
            if len(seq) < 2:
                continue
            # Окна: для каждой позиции t (от 1 до len-1) берём
            # input = seq[max(0, t-window):t], target = seq[t]
            for t in range(1, len(seq)):
                inp = seq[max(0, t - window):t]
                tgt = seq[t]
                self.samples.append((inp, tgt))

        print(f"  Создано обучающих примеров: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        if len(inp) < self.window:
            pad = [PAD_IDX] * (self.window - len(inp))
            inp = pad + inp
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


def load_split(name: str) -> list[list[int]]:
    """Загружает последовательности одного сета (train/val/test)."""
    path = SPLITS_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df["sequence"].tolist()


def build_datasets(window: int = WINDOW):
    """Собирает три датасета."""
    print("Читаю splits...")
    train_seq = load_split("train")
    val_seq   = load_split("val")
    test_seq  = load_split("test")

    print(f"Train: {len(train_seq):,} последовательностей")
    train_ds = SequenceDataset(train_seq, window)

    print(f"Val:   {len(val_seq):,} последовательностей")
    val_ds = SequenceDataset(val_seq, window)

    print(f"Test:  {len(test_seq):,} последовательностей")
    test_ds = SequenceDataset(test_seq, window)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = build_datasets()

    print(f"\nРазмеры:")
    print(f"  train: {len(train_ds):,}")
    print(f"  val:   {len(val_ds):,}")
    print(f"  test:  {len(test_ds):,}")

    x, y = train_ds[0]
    print(f"\nПример:")
    print(f"  input shape: {x.shape}")
    print(f"  input:       {x.tolist()}")
    print(f"  target:      {y.item()}")