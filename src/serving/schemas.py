"""
Pydantic-схемы запроса/ответа для /recommend.
"""
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    history: list[str] = Field(
        ...,
        description="История покупок клиента (StockCode) от старых к новым",
        min_length=1,
    )
    top_k: int = Field(10, ge=1, le=100, description="Сколько рекомендаций вернуть")


class RecommendItem(BaseModel):
    stockcode: str
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[RecommendItem]
    unknown_items: list[str] = Field(
        default_factory=list,
        description="StockCode из запроса, которых нет в словаре модели",
    )


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()} 
    status: str
    model_loaded: bool
    vocab_size: int
    window: int