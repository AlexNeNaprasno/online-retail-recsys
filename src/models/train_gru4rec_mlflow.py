"""
GRU4Rec training с MLflow tracking.

Логируем:
  - параметры (batch, lr, epochs, emb_dim, hidden_dim, dropout, window)
  - метрики по эпохам (train_loss, val_loss)
  - финальные метрики (HitRate@10, NDCG@10, MRR@10 на val/test)
  - артефакты (чекпоинт, история, график кривой обучения)
"""
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dataset import build_datasets, load_split, WINDOW, PAD_IDX
from src.models.gru4rec import GRU4Rec, load_vocab_size

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = CKPT_DIR / "gru4rec_best.pt"

# ==== Гиперпараметры ====
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 20
PATIENCE = 3
EMB_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.2
NUM_WORKERS = 0
SEED = 42

EXPERIMENT_NAME = "online-retail-recsys"
RUN_NAME = f"gru4rec_emb{EMB_DIM}_hid{HIDDEN_DIM}_lr{LR}"

def _safe_metric_name(name: str) -> str:
    """MLflow запрещает '@' в именах метрик."""
    return name.replace("@", "_at_")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def seq_to_input(seq, window: int = WINDOW) -> list[int]:
    seq = list(seq)
    inp = seq[-window:] if len(seq) >= window else seq
    if len(inp) < window:
        inp = [PAD_IDX] * (window - len(inp)) + inp
    return inp


@torch.no_grad()
def predict_topk(model, inputs, k, device, batch_size=512):
    x = torch.tensor(inputs, dtype=torch.long)
    topk_all = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i+batch_size].to(device)
        logits = model(xb)
        topk = logits.topk(k, dim=-1).indices
        topk_all.append(topk.cpu())
    return torch.cat(topk_all, dim=0).numpy()


def eval_metrics(model, split_name, k, device):
    seqs = load_split(split_name)
    inputs = [seq_to_input(s[:-1]) for s in seqs]
    targets = [int(s[-1]) for s in seqs]
    topk = predict_topk(model, inputs, k, device)

    hits, ndcgs, mrrs = [], [], []
    for row, tgt in zip(topk, targets):
        hits.append(1.0 if tgt in row[:k] else 0.0)
        ndcg = 0.0
        for i, p in enumerate(row[:k], 1):
            if p == tgt:
                ndcg = 1.0 / np.log2(i + 1)
                break
        ndcgs.append(ndcg)
        mrr = 0.0
        for i, p in enumerate(row[:k], 1):
            if p == tgt:
                mrr = 1.0 / i
                break
        mrrs.append(mrr)

    return {
        f"HitRate@{k}": float(np.mean(hits)),
        f"NDCG@{k}":    float(np.mean(ndcgs)),
        f"MRR@{k}":     float(np.mean(mrrs)),
    }


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    train_ds, val_ds, _ = build_datasets(window=WINDOW)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    vocab_size = load_vocab_size()

    with mlflow.start_run(run_name=RUN_NAME) as run:
        print(f"MLflow run_id: {run.info.run_id}")

        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "window": WINDOW,
            "vocab_size": vocab_size,
            "seed": SEED,
        })

        model = GRU4Rec(
            vocab_size=vocab_size,
            emb_dim=EMB_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("n_params", n_params)
        print(f"Параметров: {n_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_loss = float("inf")
        epochs_without_improve = 0
        history_train, history_val = [], []

        print(f"\nСтарт обучения: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}\n")

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            dt = time.time() - t0

            history_train.append(train_loss)
            history_val.append(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, step=epoch)
            mlflow.log_metric("epoch_time_sec", dt, step=epoch)

            print(f"  [epoch {epoch:02d}/{EPOCHS}] "
                  f"train={train_loss:.4f} val={val_loss:.4f} ({dt:.1f}s)")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improve = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "vocab_size": vocab_size,
                    "emb_dim": EMB_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "window": WINDOW,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                }, BEST_CKPT)
                print(f"  ✓ Best (val={best_val_loss:.4f})")
            else:
                epochs_without_improve += 1
                print(f"  ✗ No improve ({epochs_without_improve}/{PATIENCE})")
                if epochs_without_improve >= PATIENCE:
                    print(f"Early stopping @ epoch {epoch}")
                    break

        print("\nОценка на val/test...")
        ckpt = torch.load(BEST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        K = 10
        val_metrics = eval_metrics(model, "val", K, device)
        test_metrics = eval_metrics(model, "test", K, device)

        print(f"  VAL:  {val_metrics}")
        print(f"  TEST: {test_metrics}")

        mlflow.log_metrics({f"val_{_safe_metric_name(k)}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{_safe_metric_name(k)}": v for k, v in test_metrics.items()})
        mlflow.log_metric("best_val_loss", best_val_loss)

        fig_path = ROOT / "reports" / "figures" / "mlflow_training_curve.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(history_train)+1), history_train, marker="o", label="train loss")
        plt.plot(range(1, len(history_val)+1), history_val, marker="s", label="val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("GRU4Rec: training curve")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=120)
        plt.close()

        mlflow.log_artifact(str(BEST_CKPT))
        mlflow.log_artifact(str(fig_path))
        mlflow.log_dict({"val": val_metrics, "test": test_metrics}, "metrics.json")
        mlflow.log_dict({"train_loss": history_train, "val_loss": history_val}, "history.json")

        print(f"\nMLflow run_id: {run.info.run_id}")
        print(f"UI: mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    main()