"""
Строит графики для README:
  1. Кривая обучения (train/val loss).
  2. Сравнение метрик popularity vs GRU4Rec.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_curve():
    with open(RESULTS_DIR / "gru4rec_history.json", encoding="utf-8") as f:
        history = json.load(f)

    train = history["train_loss"]
    val = history["val_loss"]
    epochs = list(range(1, len(train) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, marker="o", label="train loss")
    plt.plot(epochs, val, marker="s", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyLoss")
    plt.title("GRU4Rec: кривая обучения")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = FIG_DIR / "training_curve.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Сохранено: {out}")


def plot_model_comparison():
    with open(RESULTS_DIR / "baseline_popularity.json", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(RESULTS_DIR / "gru4rec_metrics.json", encoding="utf-8") as f:
        gru = json.load(f)

    K = baseline["K"]
    metrics = [f"HitRate@{K}", f"NDCG@{K}", f"MRR@{K}"]
    pop_vals = [baseline["test"][m] for m in metrics]
    gru_vals = [gru["test"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, pop_vals, width, label="Popularity", color="#999999")
    bars2 = ax.bar(x + width/2, gru_vals, width, label="GRU4Rec", color="#2b7bba")

    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение моделей на test")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "model_comparison.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    plot_training_curve()
    plot_model_comparison()