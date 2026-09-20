"""Тесты Pydantic-схем FastAPI-сервиса."""
import pytest
from pydantic import ValidationError

from src.serving.schemas import RecommendRequest, RecommendResponse, RecommendItem


def test_recommend_request_valid():
    req = RecommendRequest(history=["A", "B", "C"], top_k=5)
    assert req.history == ["A", "B", "C"]
    assert req.top_k == 5


def test_recommend_request_default_topk():
    req = RecommendRequest(history=["A"])
    assert req.top_k == 10


def test_recommend_request_rejects_empty_history():
    with pytest.raises(ValidationError):
        RecommendRequest(history=[], top_k=5)


def test_recommend_request_rejects_bad_topk():
    with pytest.raises(ValidationError):
        RecommendRequest(history=["A"], top_k=0)
    with pytest.raises(ValidationError):
        RecommendRequest(history=["A"], top_k=1000)


def test_recommend_response_serialization():
    resp = RecommendResponse(
        recommendations=[RecommendItem(stockcode="85123A", score=0.5)],
        unknown_items=["FOO"],
    )
    data = resp.model_dump()
    assert data["recommendations"][0]["stockcode"] == "85123A"
    assert data["unknown_items"] == ["FOO"]