"""Детерминированная бизнес-логика агроскоринга."""

from fastapi import HTTPException, status

from .schemas import FarmRequest, RiskLevel


ALLOWED_REGIONS = frozenset({"Krasnodar", "Rostov", "Stavropol"})


def calculate_risk(data: FarmRequest) -> float:
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


def get_risk_level(score: float) -> RiskLevel:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def get_recommendation(level: RiskLevel) -> str:
    return {
        "low": "Стандартное рассмотрение",
        "medium": "Требуется дополнительная проверка",
        "high": "Высокий риск. Требуется ручное рассмотрение",
    }[level]


def validate_business_rules(data: FarmRequest) -> None:
    if data.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown region: {data.region}. Allowed regions: {sorted(ALLOWED_REGIONS)}",
        )


def make_prediction_result(data: FarmRequest, score: float, request_id: str, version: str) -> dict:
    level = get_risk_level(score)
    return {
        "request_id": request_id,
        "farm_id": data.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": version,
    }


def build_scoring_explanation_prompt(prediction: dict) -> str:
    return (
        "Сформируй краткое объяснение для оператора банка по результату "
        "алгоритмического агроскоринга.\n\n"
        f"risk_score: {prediction['risk_score']}\n"
        f"risk_level: {prediction['risk_level']}\n"
        f"базовая рекомендация: {prediction['recommendation']}\n"
        f"версия скоринговой модели: {prediction['model_version']}\n\n"
        "Ответь в 2–4 предложениях. Не меняй категорию риска, не придумывай "
        "факторы и укажи на необходимость применения правил банка."
    )

