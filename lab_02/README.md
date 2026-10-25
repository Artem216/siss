# Лабораторная работа №2 — расширенный Agro Scoring API

Асинхронный FastAPI-сервис с SQLite, сквозным `request_id`, explicit/dynamic
batching, backpressure, пагинацией и интеграцией с Vireonix LLM.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn agro_api.main:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Health check: <http://127.0.0.1:8000/health>

Путь к SQLite можно изменить переменной `AGRO_DATABASE_PATH`, а адрес
LLM-провайдера — переменной `LLM_BASE_URL`.

## Тестирование

```bash
pytest -q
```

Тесты используют временную SQLite и mock transport для LLM, поэтому не требуют
доступа к внешней сети.

## Отчёт

Основной отчёт: [report.tex](report.tex). Дополнительная версия:
[report.md](report.md).

```bash
latexmk -pdf report.tex
```

