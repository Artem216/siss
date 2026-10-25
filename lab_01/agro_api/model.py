"""Имитация ML-модели агроскоринга."""

from .schemas import FarmRequest


def calculate_risk(data: FarmRequest) -> float:
    """Рассчитать условную оценку риска в диапазоне от 0 до 1."""

    score = 0.1
    if data.payment_delay_days > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    return round(min(score, 1.0), 2)

