"""Локальный OpenAI-совместимый провайдер для воспроизводимых скриншотов."""

from fastapi import FastAPI


app = FastAPI(title="Demo LLM Provider")


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict):
    return {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "model": "demo-agro-explainer",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "Результат указывает на высокий уровень риска. "
                        "Следует применить правила банка и передать заявку "
                        "оператору для дополнительной проверки."
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 24, "total_tokens": 66},
    }

