"""
GRU4Rec: базовая DL-модель для next-item recommendation.

Архитектура:
  Embedding -> Dropout -> GRU -> Dropout -> Linear -> logits по всем товарам.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = ROOT / "data" / "processed" / "vocab.json"


class GRU4Rec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window)
        e = self.emb(x)                 
        e = self.emb_dropout(e)

        out, _ = self.gru(e)            
        last = out[:, -1, :]            

        last = self.dropout(last)
        logits = self.fc(last)          
        return logits


def load_vocab_size() -> int:
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab)


if __name__ == "__main__":
    vocab_size = load_vocab_size()
    print(f"vocab_size = {vocab_size}")

    model = GRU4Rec(vocab_size)
    print(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Обучаемых параметров: {n_params:,}")

    batch_size = 4
    window = 20
    x = torch.randint(1, vocab_size, (batch_size, window))  # 0 не берём (это паддинг)
    logits = model(x)

    print(f"\nx.shape      = {tuple(x.shape)}")
    print(f"logits.shape = {tuple(logits.shape)}")

    criterion = nn.CrossEntropyLoss()
    y = torch.randint(1, vocab_size, (batch_size,))
    loss = criterion(logits, y)
    print(f"loss (random) = {loss.item():.4f}")