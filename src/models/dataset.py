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
from torch.utils.data import Dataset, DataLoader

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
            seq = list(seq)                 # numpy -> list ОДИН РАЗ
            for t in range(1, len(seq)):
                inp = seq[max(0, t - window):t]
                tgt = seq[t]
                self.samples.append((inp, tgt))

        print(f"  Создано обучающих примеров: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        inp = list(inp)                     # страховка
        if len(inp) < self.window:
            pad = [PAD_IDX] * (self.window - len(inp))
            inp = pad + inp
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


def load_split(name: str) -> list[list[int]]:
    path = SPLITS_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    return df["sequence"].tolist()


def build_datasets(window: int = WINDOW):
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
    print(f"\nПример train_ds[0]:")
    print(f"  input shape: {tuple(x.shape)}")
    print(f"  input:       {x.tolist()}")
    print(f"  target:      {y.item()}")

    print("\nПроверка форм:")
    for i in [0, 1, 2, 10, 100, 1000, 10000]:
        x, y = train_ds[i]
        print(f"  train_ds[{i}]: x.shape={tuple(x.shape)}, y={y.item()}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    xb, yb = next(iter(loader))
    print(f"\nБатч: x={tuple(xb.shape)}, y={tuple(yb.shape)}")
    print(f"x_batch[0] = {xb[0].tolist()}")
    print(f"y_batch    = {yb.tolist()}")