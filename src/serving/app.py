"""
FastAPI-сервис рекомендаций.

Эндпоинты:
  GET  /health    — проверка живости
  POST /recommend — топ-K рекомендаций по истории покупок
  GET  /          — корень, краткая информация
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.serving.model_loader import RecommenderService
from src.serving.schemas import (
    HealthResponse,
    RecommendItem,
    RecommendRequest,
    RecommendResponse,
)

# Глобальный сервис, создаётся один раз при старте
service: RecommenderService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем модель при старте, освобождаем — при остановке."""
    global service
    print("Старт: загружаю модель...")
    service = RecommenderService()
    yield
    print("Остановка: освобождаю ресурсы")
    service = None


app = FastAPI(
    title="Online Retail RecSys API",
    description="GRU4Rec-based next-item recommendations",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "online-retail-recsys",
        "model": "gru4rec",
        "endpoints": ["/health", "/recommend", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_loaded=True,
        vocab_size=len(service.vocab),
        window=service.window,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["recsys"])
def recommend(req: RecommendRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    recs, unknown = service.recommend(req.history, top_k=req.top_k)

    if not recs:
        raise HTTPException(
            status_code=422,
            detail="No recommendations produced. Check that history contains valid StockCodes.",
        )

    return RecommendResponse(
        recommendations=[RecommendItem(stockcode=c, score=s) for c, s in recs],
        unknown_items=unknown,
    )