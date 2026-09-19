"""
Загрузка обученного GRU4Rec и словаря; методы inference.
Сервис создаётся один раз при старте приложения.
"""
import json
from pathlib import Path

import torch

from src.models.gru4rec import GRU4Rec

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = ROOT / "checkpoints" / "gru4rec_best.pt"
DEFAULT_VOCAB = ROOT / "data" / "processed" / "vocab.json"

PAD_IDX = 0


class RecommenderService:
    def __init__(
        self,
        ckpt_path: Path = DEFAULT_CKPT,
        vocab_path: Path = DEFAULT_VOCAB,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Словарь StockCode -> index
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        # Обратный index -> StockCode
        self.inv_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Чекпоинт
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.window: int = ckpt.get("window", 20)
        self.model = GRU4Rec(
            vocab_size=ckpt["vocab_size"],
            emb_dim=ckpt["emb_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        print(f"[RecommenderService] device={self.device}, "
              f"vocab={len(self.vocab)}, window={self.window}, "
              f"ckpt_epoch={ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss'):.4f}")

    def _codes_to_indices(self, history: list[str]) -> tuple[list[int], list[str]]:
        """StockCode -> индексы. Неизвестные заменяются на <UNK> и возвращаются отдельно."""
        unk_idx = self.vocab.get("<UNK>", PAD_IDX)
        indices: list[int] = []
        unknown: list[str] = []
        for code in history:
            idx = self.vocab.get(code)
            if idx is None:
                unknown.append(code)
                idx = unk_idx
            indices.append(idx)
        return indices, unknown

    def _pad_or_truncate(self, indices: list[int]) -> list[int]:
        if len(indices) >= self.window:
            return indices[-self.window:]
        return [PAD_IDX] * (self.window - len(indices)) + indices

    @torch.no_grad()
    def recommend(self, history: list[str], top_k: int = 10) -> tuple[list[tuple[str, float]], list[str]]:
        indices, unknown = self._codes_to_indices(history)
        if not indices:
            return [], unknown

        windowed = self._pad_or_truncate(indices)
        x = torch.tensor([windowed], dtype=torch.long, device=self.device)

        logits = self.model(x)                       # (1, vocab)
        probs = torch.softmax(logits, dim=-1)        # (1, vocab)
        top = probs.topk(top_k, dim=-1)              # values, indices

        recommendations: list[tuple[str, float]] = []
        for score, idx in zip(top.values[0].cpu().tolist(), top.indices[0].cpu().tolist()):
            code = self.inv_vocab.get(int(idx))
            # Пропускаем <UNK> и PAD
            if code is None or code == "<UNK>" or int(idx) == PAD_IDX:
                continue
            recommendations.append((code, float(score)))

        return recommendations, unknown