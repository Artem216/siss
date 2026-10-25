"""Асинхронная имитация единственного вычислительного ресурса модели."""

import asyncio

from .config import Settings
from .schemas import FarmRequest
from .services import calculate_risk


class ModelEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = asyncio.Lock()

    async def predict_single(self, data: FarmRequest) -> float:
        async with self.lock:
            await asyncio.sleep(
                self.settings.model_fixed_overhead_seconds
                + self.settings.model_per_item_seconds
            )
            return calculate_risk(data)

    async def predict_batch(self, items: list[FarmRequest]) -> list[float]:
        async with self.lock:
            await asyncio.sleep(
                self.settings.model_fixed_overhead_seconds
                + self.settings.model_per_item_seconds * len(items)
            )
            return [calculate_risk(item) for item in items]

