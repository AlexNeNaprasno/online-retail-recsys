"""Тесты PyTorch-датасета для GRU4Rec."""
import torch

from src.models.dataset import SequenceDataset, PAD_IDX


def test_dataset_produces_correct_shapes():
    seqs = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]
    ds = SequenceDataset(seqs, window=5)
    x, y = ds[0]
    assert x.shape == (5,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_pads_short_sequences_on_the_left():
    seqs = [[10, 20, 30]]
    ds = SequenceDataset(seqs, window=5)
    x, y = ds[0]
    assert x.tolist() == [0, 0, 0, 0, 10]
    assert y.item() == 20


def test_dataset_skips_singleton_sequences():
    seqs = [[1], [2], [3]]
    ds = SequenceDataset(seqs, window=5)
    assert len(ds) == 0


def test_dataset_sliding_window_count():
    seqs = [[1, 2, 3, 4, 5]]
    ds = SequenceDataset(seqs, window=3)
    assert len(ds) == 4


def test_pad_idx_is_zero():
    assert PAD_IDX == 0