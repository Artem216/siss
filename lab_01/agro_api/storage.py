"""Временное in-memory хранилище прогнозов."""

from typing import Any


predictions: dict[str, dict[str, Any]] = {}


def save_prediction(request_id: str, result: dict[str, Any]) -> None:
    predictions[request_id] = result


def get_prediction(request_id: str) -> dict[str, Any] | None:
    return predictions.get(request_id)


def list_predictions() -> list[dict[str, Any]]:
    return list(predictions.values())


def clear_predictions() -> None:
    """Очистить хранилище между тестами."""

    predictions.clear()

