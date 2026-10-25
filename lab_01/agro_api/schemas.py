"""Pydantic-схемы входных данных и ответов API."""

from typing import Literal

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]


class FarmRequest(BaseModel):
    """Данные хозяйства для оценки риска."""

    farm_id: str = Field(min_length=1, description="Идентификатор хозяйства")
    region: str = Field(min_length=1, description="Регион хозяйства")
    crop_type: str = Field(min_length=1, description="Основная культура")
    area_ha: float = Field(gt=0, description="Площадь посевов, га")
    temperature_avg: float = Field(ge=-60, le=60, description="Средняя температура, °C")
    precipitation_mm: float = Field(ge=0, description="Количество осадков, мм")
    payment_delay_days: int = Field(ge=0, description="Дней просрочки платежа")
    previous_defaults: int = Field(ge=0, description="Количество предыдущих дефолтов")
    debt: float = Field(ge=0, description="Текущая задолженность")


class PredictionResponse(BaseModel):
    """Результат оценки риска."""

    request_id: str
    farm_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    """Состояние API."""

    status: Literal["ok"]


class ModelInfoResponse(BaseModel):
    """Метаданные и состояние модели."""

    model_name: str
    model_version: str
    model_type: str
    status: Literal["ready", "unavailable"]

