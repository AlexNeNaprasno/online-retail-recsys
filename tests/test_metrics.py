"""Тесты метрик ранжирования."""
import numpy as np

from src.models.baseline_popularity import hit_rate_at_k, ndcg_at_k, mrr_at_k


def test_hit_rate_hit():
    assert hit_rate_at_k([1, 2, 3, 4, 5], 3, 5) == 1.0


def test_hit_rate_miss():
    assert hit_rate_at_k([1, 2, 3], 99, 3) == 0.0


def test_hit_rate_respects_k():
    assert hit_rate_at_k([1, 2, 3, 4, 5, 99], 99, 5) == 0.0


def test_ndcg_at_first_position():
    assert ndcg_at_k([5, 1, 2, 3], 5, 4) == 1.0


def test_ndcg_at_second_position():
    expected = 1.0 / np.log2(3)
    assert abs(ndcg_at_k([0, 5, 1, 2], 5, 4) - expected) < 1e-9


def test_ndcg_miss():
    assert ndcg_at_k([1, 2, 3], 99, 3) == 0.0


def test_mrr_first_position():
    assert mrr_at_k([5, 1, 2], 5, 3) == 1.0


def test_mrr_third_position():
    assert abs(mrr_at_k([1, 2, 5], 5, 3) - 1 / 3) < 1e-9


def test_mrr_miss():
    assert mrr_at_k([1, 2, 3], 99, 3) == 0.0