"""HTTP-слой REST API сервиса агроскоринга."""

import logging
import time
import uuid
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, status

from .model import calculate_risk
from .schemas import FarmRequest, HealthResponse, ModelInfoResponse, PredictionResponse, RiskLevel
from .services import ALLOWED_REGIONS, get_recommendation, get_risk_level, is_region_allowed
from .storage import get_prediction as load_prediction
from .storage import list_predictions, save_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"
MODEL_READY = True

app = FastAPI(
    title="Agro Scoring API",
    description="REST API для оценки риска сельскохозяйственных предприятий.",
    version="1.0.0",
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.perf_counter() - start_time, 6))
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description="Подтверждает, что HTTP API запущен и отвечает.",
)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description="Возвращает название, версию, тип и текущее состояние модели.",
)
def model_info() -> dict[str, str]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": "ready" if MODEL_READY else "unavailable",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Оценить риск хозяйства",
    description="Проверяет данные, выполняет инференс и сохраняет результат.",
    responses={400: {"description": "Неизвестный регион"}, 503: {"description": "Модель недоступна"}},
)
def predict(request: FarmRequest) -> dict[str, object]:
    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is temporarily unavailable",
        )
    if not is_region_allowed(request.region):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown region: {request.region}. Allowed regions: {sorted(ALLOWED_REGIONS)}",
        )

    logger.info("Prediction request received | farm_id=%s", request.farm_id)
    score = calculate_risk(request)
    level = get_risk_level(score)
    request_id = str(uuid.uuid4())
    result: dict[str, object] = {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": MODEL_VERSION,
    }
    save_prediction(request_id, result)
    logger.info(
        "Prediction completed | request_id=%s | farm_id=%s | risk_score=%s | risk_level=%s",
        request_id,
        request.farm_id,
        score,
        level,
    )
    return result


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="Получить список прогнозов",
    description="Возвращает прогнозы с ограничением количества и фильтрацией по уровню риска.",
)
def get_predictions(
    limit: Annotated[int, Query(ge=1, le=100, description="Максимальное количество результатов")] = 10,
    risk_level: Annotated[RiskLevel | None, Query(description="Фильтр: low, medium или high")] = None,
) -> list[dict[str, object]]:
    values = list_predictions()
    if risk_level is not None:
        values = [item for item in values if item["risk_level"] == risk_level]
    return values[:limit]


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить прогноз по request_id",
    description="Возвращает ранее сохранённый прогноз по его уникальному идентификатору.",
    responses={404: {"description": "Прогноз не найден"}},
)
def get_prediction(request_id: str) -> dict[str, object]:
    result = load_prediction(request_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return result

