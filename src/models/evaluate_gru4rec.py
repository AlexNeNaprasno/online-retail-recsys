"""
Оценка обученного GRU4Rec на val и test.

Метрики: HitRate@K, NDCG@K, MRR@K.
Сравнение с popularity baseline.
"""
import json
from pathlib import Path

import numpy as np
import torch

from src.models.dataset import load_split, WINDOW, PAD_IDX
from src.models.gru4rec import GRU4Rec

ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "checkpoints" / "gru4rec_best.pt"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = GRU4Rec(
        vocab_size=ckpt["vocab_size"],
        emb_dim=ckpt["emb_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def seq_to_input(seq, window: int = WINDOW) -> list[int]:
    seq = list(seq)
    inp = seq[-window:] if len(seq) >= window else seq
    if len(inp) < window:
        inp = [PAD_IDX] * (window - len(inp)) + inp
    return inp


@torch.no_grad()
def predict_topk(model, inputs: list[list[int]], k: int, device, batch_size: int = 512) -> np.ndarray:
    x = torch.tensor(inputs, dtype=torch.long)
    topk_all = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i+batch_size].to(device)
        logits = model(xb)                      # (B, vocab)
        topk = logits.topk(k, dim=-1).indices   # (B, k)
        topk_all.append(topk.cpu())
    return torch.cat(topk_all, dim=0).numpy()


def hit_rate_at_k(preds: np.ndarray, target: int, k: int) -> float:
    return 1.0 if target in preds[:k] else 0.0


def ndcg_at_k(preds: np.ndarray, target: int, k: int) -> float:
    for i, p in enumerate(preds[:k], start=1):
        if p == target:
            return 1.0 / np.log2(i + 1)
    return 0.0


def mrr_at_k(preds: np.ndarray, target: int, k: int) -> float:
    for i, p in enumerate(preds[:k], start=1):
        if p == target:
            return 1.0 / i
    return 0.0


def evaluate_split(model, split_name: str, k: int, device) -> dict:
    seqs = load_split(split_name)
    inputs = [seq_to_input(s[:-1]) for s in seqs]      
    targets = [int(s[-1]) for s in seqs]               

    topk = predict_topk(model, inputs, k, device)

    hits, ndcgs, mrrs = [], [], []
    for pred_row, tgt in zip(topk, targets):
        hits.append(hit_rate_at_k(pred_row, tgt, k))
        ndcgs.append(ndcg_at_k(pred_row, tgt, k))
        mrrs.append(mrr_at_k(pred_row, tgt, k))

    return {
        "n_users": len(hits),
        f"HitRate@{k}": float(np.mean(hits)),
        f"NDCG@{k}":    float(np.mean(ndcgs)),
        f"MRR@{k}":     float(np.mean(mrrs)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, ckpt = load_model(device)
    print(f"Чекпоинт: epoch={ckpt['epoch']}, best_val_loss={ckpt['best_val_loss']:.4f}")

    K = 10
    print(f"\nОценка с K={K}...\n")

    val_metrics = evaluate_split(model, "val", K, device)
    test_metrics = evaluate_split(model, "test", K, device)

    print("=== GRU4Rec ===")
    print("VAL:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")
    print("TEST:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    baseline_path = RESULTS_DIR / "baseline_popularity.json"
    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        print("\n=== Сравнение с popularity baseline (test) ===")
        print(f"{'Метрика':<12} {'Popularity':>12} {'GRU4Rec':>12} {'Δ %':>10}")
        for metric in [f"HitRate@{K}", f"NDCG@{K}", f"MRR@{K}"]:
            b = baseline["test"][metric]
            g = test_metrics[metric]
            delta = (g - b) / b * 100 if b > 0 else 0.0
            print(f"{metric:<12} {b:>12.4f} {g:>12.4f} {delta:>+9.1f}%")

    results = {
        "model": "gru4rec",
        "K": K,
        "val": val_metrics,
        "test": test_metrics,
    }
    out = RESULTS_DIR / "gru4rec_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nСохранил: {out}")


if __name__ == "__main__":
    main()