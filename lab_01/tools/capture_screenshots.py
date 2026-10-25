"""Создание скриншотов отчёта из запущенного Swagger UI."""

import json
import os
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


BASE_URL = os.getenv("AGRO_API_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = Path(__file__).parents[1] / "docs" / "screenshots"

VALID_PAYLOAD = {
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


def open_operation(page: Page, operation_id: str) -> None:
    operation = page.locator(f"#{operation_id}")
    operation.locator(".opblock-summary").click()
    operation.locator(".try-out__btn").click()


def execute_post(page: Page, payload: dict[str, object], filename: str) -> None:
    page.goto(f"{BASE_URL}/docs", wait_until="networkidle")
    open_operation(page, "operations-default-predict_predict_post")
    operation = page.locator("#operations-default-predict_predict_post")
    operation.locator("textarea").fill(json.dumps(payload, ensure_ascii=False, indent=2))
    operation.locator(".execute").click()
    operation.locator(".live-responses-table td.response-col_status:not(.col_header)").wait_for()
    operation.screenshot(path=str(OUTPUT_DIR / filename))


def execute_404(page: Page) -> None:
    page.goto(f"{BASE_URL}/docs", wait_until="networkidle")
    open_operation(page, "operations-default-get_prediction_predictions__request_id__get")
    operation = page.locator("#operations-default-get_prediction_predictions__request_id__get")
    operation.locator("input").fill("missing-id")
    operation.locator(".execute").click()
    operation.locator(".live-responses-table .response-col_status:has-text('404')").wait_for()
    operation.screenshot(path=str(OUTPUT_DIR / "404-not-found.png"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(f"{BASE_URL}/docs", wait_until="networkidle")
        page.screenshot(path=str(OUTPUT_DIR / "swagger-ui.png"), full_page=True)
        execute_post(page, VALID_PAYLOAD, "successful-predict.png")
        invalid_payload = VALID_PAYLOAD | {"area_ha": -100}
        execute_post(page, invalid_payload, "pydantic-error.png")
        execute_404(page)
        browser.close()


if __name__ == "__main__":
    main()
