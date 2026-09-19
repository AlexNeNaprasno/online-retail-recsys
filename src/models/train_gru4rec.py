"""
Training loop для GRU4Rec с early stopping.

- Обучаем на train.
- Считаем val loss каждую эпоху.
- Сохраняем лучший чекпоинт по val loss.
- Early stopping по patience.
- Логируем прогресс в консоль.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dataset import build_datasets, WINDOW
from src.models.gru4rec import GRU4Rec, load_vocab_size

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = CKPT_DIR / "gru4rec_best.pt"
HISTORY_PATH = ROOT / "results" / "gru4rec_history.json"
HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==== Гиперпараметры ====
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 20
PATIENCE = 3               # сколько эпох ждать улучшения
EMB_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.2
NUM_WORKERS = 0            
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    dt = time.time() - t0
    print(f"  [epoch {epoch:02d}/{total_epochs}] train loss = {avg_loss:.4f}  ({dt:.1f}s)")
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Данные
    train_ds, val_ds, _ = build_datasets(window=WINDOW)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # Модель
    vocab_size = load_vocab_size()
    print(f"vocab_size = {vocab_size}")

    model = GRU4Rec(
        vocab_size=vocab_size,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Параметров: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improve = 0

    print(f"\nСтарт обучения: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, EPOCHS
        )
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"  [epoch {epoch:02d}/{EPOCHS}] val   loss = {val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "emb_dim": EMB_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "window": WINDOW,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                BEST_CKPT,
            )
            print(f"  ✓ Лучший чекпоинт сохранён (val loss = {best_val_loss:.4f}) → {BEST_CKPT.name}")
        else:
            epochs_without_improve += 1
            print(f"  ✗ Нет улучшения ({epochs_without_improve}/{PATIENCE})")

        if epochs_without_improve >= PATIENCE:
            print(f"\nEarly stopping на эпохе {epoch}. Лучший val loss = {best_val_loss:.4f}")
            break

        print()

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nИстория сохранена: {HISTORY_PATH}")
    print(f"Лучший чекпоинт: {BEST_CKPT}")


if __name__ == "__main__":
    main()