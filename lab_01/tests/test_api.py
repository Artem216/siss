"""Автоматические тесты REST API."""

import uuid

import pytest
from fastapi.testclient import TestClient

from agro_api import main
from agro_api.services import get_risk_level
from agro_api.storage import clear_predictions


client = TestClient(main.app)


@pytest.fixture(autouse=True)
def reset_application_state(monkeypatch):
    clear_predictions()
    monkeypatch.setattr(main, "MODEL_READY", True)
    yield
    clear_predictions()


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return {
        "farm_id": "FARM-001",
        "region": "Krasnodar",
        "crop_type": "wheat",
        "area_ha": 2500,
        "temperature_avg": 24.3,
        "precipitation_mm": 320,
        "payment_delay_days": 45,
        "previous_defaults": 1,
        "debt": 6_500_000,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert float(response.headers["X-Process-Time"]) >= 0


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "agro-risk-model",
        "model_version": "1.0",
        "model_type": "risk-scoring",
        "status": "ready",
    }


def test_successful_prediction(valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert body["farm_id"] == "FARM-001"
    assert body["risk_score"] == 0.9
    assert body["risk_level"] == "high"
    assert body["model_version"] == "1.0"
    assert uuid.UUID(body["request_id"]).version == 4


def test_negative_area_returns_422(valid_payload):
    valid_payload["area_ha"] = -100
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "area_ha"


def test_unknown_region_returns_400(valid_payload):
    valid_payload["region"] = "Unknown"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 400
    assert "Unknown region" in response.json()["detail"]


def test_get_existing_prediction(valid_payload):
    created = client.post("/predict", json=valid_payload).json()
    response = client.get(f"/predictions/{created['request_id']}")
    assert response.status_code == 200
    assert response.json() == created


def test_get_unknown_prediction_returns_404():
    response = client.get("/predictions/missing-id")
    assert response.status_code == 404
    assert response.json() == {"detail": "Prediction not found"}


def test_limit_query_parameter(valid_payload):
    for index in range(3):
        valid_payload["farm_id"] = f"FARM-{index}"
        assert client.post("/predict", json=valid_payload).status_code == 200
    response = client.get("/predictions?limit=2")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_risk_level_filter(valid_payload):
    low_payload = valid_payload.copy()
    low_payload.update(
        farm_id="FARM-LOW",
        payment_delay_days=0,
        previous_defaults=0,
        debt=0,
        precipitation_mm=300,
    )
    client.post("/predict", json=valid_payload)
    client.post("/predict", json=low_payload)
    response = client.get("/predictions?risk_level=high")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["risk_level"] == "high"


def test_negative_limit_returns_422():
    response = client.get("/predictions?limit=-5")
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.0, "low"), (0.29, "low"), (0.3, "medium"), (0.69, "medium"), (0.7, "high"), (1.0, "high")],
)
def test_risk_level_boundaries(score, expected):
    assert get_risk_level(score) == expected


def test_invalid_risk_level_returns_422():
    response = client.get("/predictions?risk_level=critical")
    assert response.status_code == 422


def test_unavailable_model_returns_503(valid_payload, monkeypatch):
    monkeypatch.setattr(main, "MODEL_READY", False)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert response.json() == {"detail": "Model is temporarily unavailable"}


def test_openapi_contract_contains_routes_and_schemas():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    document = response.json()
    expected_paths = {"/health", "/model-info", "/predict", "/predictions", "/predictions/{request_id}"}
    assert expected_paths <= set(document["paths"])
    schemas = document["components"]["schemas"]
    assert "FarmRequest" in schemas
    assert "PredictionResponse" in schemas


def test_swagger_ui_is_available():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()

