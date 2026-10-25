"""Фабрика FastAPI-приложения и публичные HTTP endpoints."""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .batching import DynamicBatcher
from .config import Settings, WEEKDAY_MAP
from .database import PredictionDatabase
from .llm import LLMService
from .model import ModelEngine
from .schemas import (
    BatchPredictionResponse,
    BatchRequest,
    FarmRequest,
    HealthResponse,
    LLMChatRequest,
    LLMChatResponse,
    ModelInfoResponse,
    PaginatedPredictionsResponse,
    PredictionResponse,
    PredictWithLLMResponse,
    QueueStatusResponse,
)
from .services import (
    build_scoring_explanation_prompt,
    get_recommendation,
    get_risk_level,
    make_prediction_result,
    validate_business_rules,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = PredictionDatabase(settings)
        await database.initialize()
        engine = ModelEngine(settings)
        batcher = DynamicBatcher(settings, engine)
        llm_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_timeout_seconds),
            transport=llm_transport,
        )
        app.state.database = database
        app.state.engine = engine
        app.state.batcher = batcher
        app.state.llm_client = llm_client
        app.state.llm = LLMService(settings, llm_client)
        batcher.start()
        logger.info("Application started | database=%s", settings.database_path)
        try:
            yield
        finally:
            await batcher.stop()
            await llm_client.aclose()

    app = FastAPI(
        title="Agro Scoring API",
        description=(
            "Расширенный AI-сервис: request_id, SQLite, explicit/dynamic "
            "batching, backpressure и интеграция с Vireonix LLM."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        logger.info(
            "Request started | request_id=%s | method=%s | path=%s",
            request_id, request.method, request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error | request_id=%s", request_id)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "request_id": request_id},
            )
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(elapsed, 6))
        logger.info(
            "Request completed | request_id=%s | status=%s | time=%.4f",
            request_id, response.status_code, elapsed,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "request_id": getattr(request.state, "request_id", None),
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": jsonable_encoder(exc.errors()),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    def require_model() -> None:
        if not settings.model_ready:
            raise HTTPException(status_code=503, detail="Model is unavailable")

    async def single_score(data: FarmRequest, request_id: str) -> dict:
        require_model()
        validate_business_rules(data)
        try:
            score = await asyncio.wait_for(
                app.state.engine.predict_single(data),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Inference timeout") from exc
        return make_prediction_result(data, score, request_id, settings.model_version)

    @app.get("/", summary="Корневая страница")
    async def root():
        return {"message": "Agro Scoring API is running", "docs": "/docs", "health": "/health"}

    @app.get("/health", response_model=HealthResponse, summary="Проверка состояния API")
    async def health():
        return {"status": "ok"}

    @app.get("/model-info", response_model=ModelInfoResponse, summary="Информация о модели")
    async def model_info():
        return {
            "model_name": settings.model_name,
            "model_version": settings.model_version,
            "model_type": settings.model_type,
            "status": "ready" if settings.model_ready else "unavailable",
        }

    @app.post("/predict", response_model=PredictionResponse, summary="Одиночный инференс")
    async def predict(data: FarmRequest, request: Request):
        result = await single_score(data, request.state.request_id)
        await app.state.database.save(result, source="single")
        return result

    @app.post(
        "/batch-predict",
        response_model=BatchPredictionResponse,
        summary="Явный пакетный инференс",
    )
    async def batch_predict(batch: BatchRequest, request: Request):
        require_model()
        for item in batch.items:
            validate_business_rules(item)
        started = time.perf_counter()
        try:
            scores = await asyncio.wait_for(
                app.state.engine.predict_batch(batch.items),
                timeout=settings.inference_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Batch inference timeout") from exc
        response_items = []
        for data, score in zip(batch.items, scores):
            level = get_risk_level(score)
            response_items.append(
                {
                    "farm_id": data.farm_id,
                    "risk_score": score,
                    "risk_level": level,
                    "recommendation": get_recommendation(level),
                    "model_version": settings.model_version,
                }
            )
        await app.state.database.save_batch(request.state.request_id, response_items)
        return {
            "request_id": request.state.request_id,
            "batch_size": len(response_items),
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
            "items": response_items,
        }

    @app.post("/predict-queued", response_model=PredictionResponse, summary="Dynamic batching")
    async def predict_queued(data: FarmRequest, request: Request):
        require_model()
        validate_business_rules(data)
        try:
            score = await app.state.batcher.submit(data, request.state.request_id)
        except asyncio.QueueFull as exc:
            raise HTTPException(
                status_code=429,
                detail="Inference queue is full. Try again later.",
                headers={"Retry-After": "1"},
            ) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Inference timeout") from exc
        result = make_prediction_result(
            data, score, request.state.request_id, settings.model_version
        )
        await app.state.database.save(result, source="queued")
        return result

    @app.get("/queue-status", response_model=QueueStatusResponse, summary="Состояние очереди")
    async def queue_status():
        return {
            "queue_size": app.state.batcher.queue.qsize(),
            "queue_max_size": settings.queue_max_size,
            "max_batch_size": settings.max_batch_size,
            "max_batch_wait_ms": settings.max_batch_wait_ms,
            "inference_timeout_seconds": settings.inference_timeout_seconds,
        }

    @app.get(
        "/predictions/by-weekday",
        response_model=PaginatedPredictionsResponse,
        summary="Пагинация прогнозов по дню недели",
    )
    async def predictions_by_weekday(
        request: Request,
        weekday: str,
        page: Annotated[int, Query(ge=1)] = 1,
        per_page: Annotated[int, Query(ge=1, le=100)] = 5,
    ):
        weekday = weekday.lower()
        if weekday not in WEEKDAY_MAP:
            raise HTTPException(
                status_code=400,
                detail="weekday must be monday, tuesday, wednesday, thursday, friday, saturday or sunday",
            )
        data, total_items, total_pages = await app.state.database.weekday_page(
            WEEKDAY_MAP[weekday], page, per_page
        )
        if (total_pages and page > total_pages) or (not total_pages and page > 1):
            raise HTTPException(status_code=404, detail="Page does not exist")
        next_page = None
        if page < total_pages:
            next_page = str(request.url.include_query_params(weekday=weekday, page=page + 1, per_page=per_page))
        prev_page = None
        if page > 1 and total_pages:
            prev_page = str(request.url.include_query_params(weekday=weekday, page=page - 1, per_page=per_page))
        return {
            "data": data,
            "meta": {
                "weekday": weekday, "current_page": page, "per_page": per_page,
                "total_pages": total_pages, "total_items": total_items,
                "next_page": next_page, "prev_page": prev_page,
            },
        }

    @app.get("/predictions", response_model=list[PredictionResponse], summary="История прогнозов")
    async def predictions(
        limit: Annotated[int, Query(ge=1, le=100)] = 10,
        risk_level: str | None = None,
    ):
        if risk_level is not None and risk_level not in {"low", "medium", "high"}:
            raise HTTPException(status_code=400, detail="risk_level must be low, medium or high")
        return await app.state.database.list_predictions(limit, risk_level)

    @app.get(
        "/requests/{request_id}/predictions",
        response_model=list[PredictionResponse],
        summary="Все прогнозы одного HTTP-запроса",
    )
    async def request_predictions(request_id: str):
        results = await app.state.database.by_request_id(request_id)
        if not results:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return results

    @app.get(
        "/predictions/{request_id}",
        response_model=PredictionResponse,
        summary="Одиночный прогноз по request_id",
    )
    async def prediction(request_id: str):
        results = await app.state.database.by_request_id(request_id)
        if not results:
            raise HTTPException(status_code=404, detail="Prediction not found")
        if len(results) > 1:
            raise HTTPException(
                status_code=409,
                detail=f"This request_id contains multiple predictions. Use /requests/{request_id}/predictions",
            )
        return results[0]

    @app.post("/demo/seed-weekdays", summary="Создать демонстрационные записи")
    async def seed_weekdays(count: Annotated[int, Query(ge=7, le=1000)] = 70):
        created = await app.state.database.seed_weekdays(count)
        return {"status": "ok", "created": created}

    @app.post("/llm/chat", response_model=LLMChatResponse, summary="Запрос к Vireonix LLM")
    async def llm_chat(data: LLMChatRequest, request: Request):
        return await app.state.llm.chat(
            data.prompt, request.state.request_id, data.model,
            data.temperature, data.max_tokens,
        )

    @app.post(
        "/predict-with-llm",
        response_model=PredictWithLLMResponse,
        summary="Агроскоринг с объяснением LLM",
    )
    async def predict_with_llm(data: FarmRequest, request: Request):
        prediction_result = await single_score(data, request.state.request_id)
        await app.state.database.save(prediction_result, source="single+llm")
        prompt = build_scoring_explanation_prompt(prediction_result)
        llm_result = await app.state.llm.chat(
            prompt, request.state.request_id, settings.llm_model,
            settings.llm_default_temperature, settings.llm_default_max_tokens,
        )
        return {"prediction": prediction_result, "llm": llm_result}

    return app
