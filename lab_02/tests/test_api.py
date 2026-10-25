"""Интеграционные тесты расширенного Agro Scoring API."""

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from agro_api.app import create_app
from agro_api.config import Settings


def llm_success(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "demo-model",
            "choices": [
                {
                    "message": {"content": "Краткое объяснение результата."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_path=tmp_path / "test.db",
        model_fixed_overhead_seconds=0.001,
        model_per_item_seconds=0.001,
        max_batch_wait_ms=5,
        inference_timeout_seconds=0.5,
        llm_timeout_seconds=0.5,
    )


@pytest.fixture
def app(settings):
    return create_app(settings, httpx.MockTransport(llm_success))


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def payload() -> dict[str, object]:
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


def assert_error(response, status_code: int) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert "detail" in body
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_health_and_observability_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert uuid.UUID(response.headers["X-Request-ID"])
    assert float(response.headers["X-Process-Time"]) >= 0


def test_client_request_id_is_preserved(client):
    response = client.get("/health", headers={"X-Request-ID": "client-id-42"})
    assert response.headers["X-Request-ID"] == "client-id-42"


def test_successful_predict_is_saved(client, payload):
    response = client.post("/predict", json=payload, headers={"X-Request-ID": "single-1"})
    assert response.status_code == 200
    assert response.json()["request_id"] == "single-1"
    assert response.json()["risk_score"] == 0.9
    history = client.get("/predictions").json()
    assert len(history) == 1
    assert history[0]["request_id"] == "single-1"


def test_negative_area_returns_422(client, payload):
    payload["area_ha"] = -100
    assert_error(client.post("/predict", json=payload), 422)


def test_unknown_region_returns_400(client, payload):
    payload["region"] = "Unknown"
    assert_error(client.post("/predict", json=payload), 400)


def test_existing_and_unknown_prediction(client, payload):
    created = client.post("/predict", json=payload).json()
    found = client.get(f"/predictions/{created['request_id']}")
    assert found.status_code == 200
    assert found.json() == created
    assert_error(client.get("/predictions/missing"), 404)


def test_explicit_batch_single_transaction_and_conflict(client, payload):
    second = payload | {"farm_id": "FARM-002"}
    response = client.post(
        "/batch-predict",
        json={"items": [payload, second]},
        headers={"X-Request-ID": "batch-1"},
    )
    assert response.status_code == 200
    assert response.json()["batch_size"] == 2
    assert_error(client.get("/predictions/batch-1"), 409)
    all_items = client.get("/requests/batch-1/predictions")
    assert all_items.status_code == 200
    assert [item["farm_id"] for item in all_items.json()] == ["FARM-001", "FARM-002"]


def test_batch_larger_than_maximum_returns_422(client, payload):
    items = [payload | {"farm_id": f"FARM-{index}"} for index in range(9)]
    assert_error(client.post("/batch-predict", json={"items": items}), 422)


def test_dynamic_batching_endpoint(client, payload):
    response = client.post("/predict-queued", json=payload)
    assert response.status_code == 200
    assert response.json()["farm_id"] == payload["farm_id"]


def test_dynamic_batching_handles_parallel_requests(client, payload):
    def send(index: int):
        return client.post("/predict-queued", json=payload | {"farm_id": f"PAR-{index}"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(send, range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert {response.json()["farm_id"] for response in responses} == {f"PAR-{i}" for i in range(4)}


def test_queue_overflow_returns_429(client, app, payload):
    app.state.batcher.submit = AsyncMock(side_effect=asyncio.QueueFull)
    response = client.post("/predict-queued", json=payload)
    assert_error(response, 429)
    assert response.headers["Retry-After"] == "1"


def test_queue_status(client):
    response = client.get("/queue-status")
    assert response.status_code == 200
    assert response.json() == {
        "queue_size": 0,
        "queue_max_size": 100,
        "max_batch_size": 8,
        "max_batch_wait_ms": 5,
        "inference_timeout_seconds": 0.5,
    }


def test_weekday_pagination(client):
    seeded = client.post("/demo/seed-weekdays?count=70")
    assert seeded.status_code == 200
    response = client.get("/predictions/by-weekday?weekday=monday&page=1&per_page=5")
    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 5
    assert body["meta"]["weekday"] == "monday"
    assert body["meta"]["total_items"] >= 5
    assert body["meta"]["next_page"] is not None


def test_invalid_page_and_weekday(client):
    assert_error(client.get("/predictions/by-weekday?weekday=monday&page=-1"), 422)
    assert_error(client.get("/predictions/by-weekday?weekday=funday"), 400)


def test_missing_page_returns_404(client):
    assert_error(client.get("/predictions/by-weekday?weekday=monday&page=2"), 404)


def test_llm_chat(client):
    response = client.post("/llm/chat", json={"prompt": "Что такое агроскоринг?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Краткое объяснение результата."
    assert response.json()["returned_model"] == "demo-model"


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (lambda request: httpx.Response(500, text="upstream failed"), 502),
        (lambda request: httpx.Response(200, text="not-json"), 502),
        (lambda request: httpx.Response(200, json={"choices": []}), 502),
        (lambda request: httpx.Response(200, json={"choices": [{"message": {}}]}), 502),
    ],
)
def test_llm_provider_errors(settings, handler, expected):
    app = create_app(settings, httpx.MockTransport(handler))
    with TestClient(app) as client:
        assert_error(client.post("/llm/chat", json={"prompt": "test"}), expected)


def test_llm_connection_error(settings):
    def unavailable(request: httpx.Request):
        raise httpx.ConnectError("offline", request=request)

    app = create_app(settings, httpx.MockTransport(unavailable))
    with TestClient(app) as client:
        assert_error(client.post("/llm/chat", json={"prompt": "test"}), 503)


def test_llm_timeout(settings):
    def timeout(request: httpx.Request):
        raise httpx.ReadTimeout("slow", request=request)

    app = create_app(settings, httpx.MockTransport(timeout))
    with TestClient(app) as client:
        assert_error(client.post("/llm/chat", json={"prompt": "test"}), 504)


def test_predict_with_llm_does_not_send_private_features(settings, payload):
    seen: dict[str, object] = {}

    def inspect_request(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return llm_success(request)

    app = create_app(settings, httpx.MockTransport(inspect_request))
    with TestClient(app) as client:
        response = client.post("/predict-with-llm", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["risk_level"] == "high"
    assert body["llm"]["answer"] == "Краткое объяснение результата."
    serialized_messages = json.dumps(seen["messages"], ensure_ascii=False)
    assert "FARM-001" not in serialized_messages
    assert "6500000" not in serialized_messages
    assert "risk_score" in serialized_messages


def test_inference_timeout(tmp_path, payload):
    settings = Settings(
        database_path=tmp_path / "timeout.db",
        inference_timeout_seconds=0.001,
        model_fixed_overhead_seconds=0.05,
        model_per_item_seconds=0,
    )
    app = create_app(settings, httpx.MockTransport(llm_success))
    with TestClient(app) as client:
        assert_error(client.post("/predict", json=payload), 504)


def test_model_unavailable(tmp_path, payload):
    settings = Settings(database_path=tmp_path / "unavailable.db", model_ready=False)
    app = create_app(settings, httpx.MockTransport(llm_success))
    with TestClient(app) as client:
        assert_error(client.post("/predict", json=payload), 503)


def test_history_filter_and_order(client, payload):
    client.post("/predict", json=payload | {"farm_id": "HIGH"})
    client.post(
        "/predict",
        json=payload
        | {
            "farm_id": "LOW",
            "payment_delay_days": 0,
            "previous_defaults": 0,
            "debt": 0,
        },
    )
    response = client.get("/predictions?risk_level=low&limit=1")
    assert response.status_code == 200
    assert response.json()[0]["farm_id"] == "LOW"
    assert_error(client.get("/predictions?risk_level=critical"), 400)


def test_openapi_contract(client):
    document = client.get("/openapi.json").json()
    expected = {
        "/health", "/model-info", "/predict", "/batch-predict",
        "/predict-queued", "/queue-status", "/predictions",
        "/predictions/by-weekday", "/requests/{request_id}/predictions",
        "/predictions/{request_id}", "/llm/chat", "/predict-with-llm",
    }
    assert expected <= set(document["paths"])
    assert "FarmRequest" in document["components"]["schemas"]


def test_unhandled_error_has_request_id(client, app, payload):
    app.state.database.save = AsyncMock(side_effect=RuntimeError("boom"))
    response = client.post("/predict", json=payload, headers={"X-Request-ID": "failure-id"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error", "request_id": "failure-id"}

