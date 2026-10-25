"""Автоматическое создание скриншотов отчёта из Swagger UI."""

import json
import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = os.getenv("AGRO_API_URL", "http://127.0.0.1:8000")
OUTPUT = Path(__file__).parents[1] / "docs" / "screenshots"

FARM = {
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


def operation(page: Page, operation_id: str):
    page.goto(f"{BASE_URL}/docs", wait_until="networkidle")
    block = page.locator(f"#{operation_id}")
    block.locator(".opblock-summary").click()
    block.locator(".try-out__btn").click()
    return block


def finish(block, filename: str) -> None:
    block.locator(".execute").click()
    block.locator(".live-responses-table td.response-col_status:not(.col_header)").wait_for()
    block.screenshot(path=str(OUTPUT / filename))


def post(page: Page, operation_id: str, body: dict, filename: str) -> None:
    block = operation(page, operation_id)
    block.locator("textarea").fill(json.dumps(body, ensure_ascii=False, indent=2))
    finish(block, filename)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(f"{BASE_URL}/docs", wait_until="networkidle")
        page.screenshot(path=str(OUTPUT / "swagger-ui.png"), full_page=True)

        post(page, "operations-default-predict_predict_post", FARM, "predict.png")

        block = operation(page, "operations-default-predictions_predictions_get")
        finish(block, "predictions-sqlite.png")

        post(
            page,
            "operations-default-batch_predict_batch_predict_post",
            {"items": [FARM, FARM | {"farm_id": "FARM-002"}]},
            "batch-predict.png",
        )

        block = operation(page, "operations-default-queue_status_queue_status_get")
        finish(block, "queue-status.png")

        page.request.post(f"{BASE_URL}/demo/seed-weekdays?count=70")
        block = operation(
            page,
            "operations-default-predictions_by_weekday_predictions_by_weekday_get",
        )
        inputs = block.locator("input")
        inputs.nth(0).fill("monday")
        inputs.nth(1).fill("1")
        inputs.nth(2).fill("5")
        finish(block, "weekday-pagination.png")

        post(
            page,
            "operations-default-llm_chat_llm_chat_post",
            {"prompt": "Объясни, что такое агроскоринг", "model": "auto", "temperature": 0.3, "max_tokens": 300},
            "llm-chat.png",
        )
        post(
            page,
            "operations-default-predict_with_llm_predict_with_llm_post",
            FARM,
            "predict-with-llm.png",
        )
        browser.close()


if __name__ == "__main__":
    main()

