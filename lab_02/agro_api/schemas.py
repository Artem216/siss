"""Pydantic-схемы публичного API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]


class FarmRequest(BaseModel):
    farm_id: str = Field(min_length=1, description="Идентификатор хозяйства")
    region: str = Field(min_length=1, description="Регион")
    crop_type: str = Field(min_length=1, description="Основная культура")
    area_ha: float = Field(gt=0, description="Площадь посевов, га")
    temperature_avg: float = Field(ge=-60, le=60, description="Средняя температура, °C")
    precipitation_mm: float = Field(ge=0, description="Количество осадков, мм")
    payment_delay_days: int = Field(ge=0, description="Дней просрочки платежа")
    previous_defaults: int = Field(ge=0, description="Количество предыдущих дефолтов")
    debt: float = Field(ge=0, description="Текущая задолженность")


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: Literal["ready", "unavailable"]


class BatchRequest(BaseModel):
    items: list[FarmRequest] = Field(min_length=1, max_length=8)


class BatchItemResponse(BaseModel):
    farm_id: str
    risk_score: float
    risk_level: RiskLevel
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[BatchItemResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictionsResponse(BaseModel):
    data: list[PredictionResponse]
    meta: PaginationMeta


class QueueStatusResponse(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


class LLMChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=5000)
    model: str = Field(default="auto", min_length=1)
    temperature: float = Field(default=0.3, ge=0, le=2)
    max_tokens: int = Field(default=300, ge=1, le=2000)


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None
    answer: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    prediction: PredictionResponse
    llm: LLMChatResponse


class ErrorResponse(BaseModel):
    detail: Any
    request_id: str | None

