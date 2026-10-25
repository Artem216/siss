"""Постобработка прогноза и бизнес-правила."""

from .schemas import RiskLevel


ALLOWED_REGIONS = frozenset({"Krasnodar", "Rostov", "Stavropol"})


def is_region_allowed(region: str) -> bool:
    return region in ALLOWED_REGIONS


def get_risk_level(score: float) -> RiskLevel:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def get_recommendation(level: RiskLevel) -> str:
    recommendations = {
        "low": "Стандартное рассмотрение",
        "medium": "Требуется дополнительная проверка",
        "high": "Высокий риск. Требуется ручное рассмотрение",
    }
    return recommendations[level]

