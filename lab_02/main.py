"""Совместимый запуск командой ``python main.py``."""

from agro_api.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agro_api.main:app", host="127.0.0.1", port=8000, reload=True)

