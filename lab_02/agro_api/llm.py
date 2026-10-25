"""Асинхронный клиент OpenAI-совместимого Vireonix API."""

import time
from typing import Any

import httpx
from fastapi import HTTPException, status

from .config import Settings


class LLMService:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client

    async def chat(
        self,
        prompt: str,
        request_id: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Ты помощник сервиса агроскоринга. Отвечай кратко. "
                        "Не изменяй переданные risk_score и risk_level и не "
                        "придумывай отсутствующие факторы."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        try:
            response = await self.client.post(
                f"{self.settings.llm_base_url}/chat/completions", json=payload
            )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM request timeout",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service is unavailable",
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if response.status_code != status.HTTP_200_OK:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"LLM provider returned an error (HTTP {response.status_code})",
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider returned invalid JSON",
            ) from exc
        choices = data.get("choices", [])
        if not choices:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM response does not contain choices",
            )
        choice = choices[0]
        message = choice.get("message", {})
        content = message.get("content")
        if not content:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM response does not contain message content",
            )
        return {
            "request_id": request_id,
            "requested_model": model,
            "returned_model": data.get("model"),
            "answer": content,
            "reasoning": message.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "latency_ms": latency_ms,
        }

