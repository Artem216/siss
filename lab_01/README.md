# Agro Scoring API

Лабораторная работа №1: REST API для оценки риска сельскохозяйственных предприятий.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn agro_api.main:app --reload
```

После запуска доступны:

- Swagger UI: <http://127.0.0.1:8000/docs>
- OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Health check: <http://127.0.0.1:8000/health>

## Тесты

```bash
pytest -q
```

Отчёт доступен в форматах [LaTeX](report.tex) и [Markdown](report.md).

Сборка LaTeX-отчёта:

```bash
latexmk -pdf report.tex
```
