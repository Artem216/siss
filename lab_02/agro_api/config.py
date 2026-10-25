"""Конфигурация приложения и значения из методических указаний."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    database_path: Path = field(
        default_factory=lambda: Path(os.getenv("AGRO_DATABASE_PATH", "agro_scoring.db"))
    )
    model_name: str = "agro-risk-model"
    model_version: str = "1.0"
    model_type: str = "risk-scoring"
    model_ready: bool = True
    max_batch_size: int = 8
    max_batch_wait_ms: int = 50
    queue_max_size: int = 100
    inference_timeout_seconds: float = 2.0
    model_fixed_overhead_seconds: float = 0.020
    model_per_item_seconds: float = 0.003
    db_busy_timeout_ms: int = 5000
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://vireonix.ai/v1")
    )
    llm_model: str = "auto"
    llm_timeout_seconds: float = 60.0
    llm_default_max_tokens: int = 300
    llm_default_temperature: float = 0.3


WEEKDAY_MAP = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6",
}

