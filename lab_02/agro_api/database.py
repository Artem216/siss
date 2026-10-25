"""Асинхронный слой хранения прогнозов в SQLite."""

import math
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
from fastapi import HTTPException, status

from .config import Settings
from .services import get_recommendation, get_risk_level


class PredictionDatabase:
    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.database_path)
        self.busy_timeout_ms = settings.db_busy_timeout_ms
        self.model_version = settings.model_version

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        db = await aiosqlite.connect(
            str(self.path), timeout=self.busy_timeout_ms / 1000
        )
        db.row_factory = aiosqlite.Row
        await db.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms};")
        try:
            yield db
        finally:
            await db.close()

    async def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.connect() as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    farm_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_request_id "
                "ON predictions(request_id);"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_risk_level "
                "ON predictions(risk_level);"
            )
            await db.commit()

    async def save(self, result: dict[str, Any], source: str) -> None:
        try:
            async with self.connect() as db:
                await db.execute(
                    """
                    INSERT INTO predictions (
                        request_id, farm_id, risk_score, risk_level,
                        recommendation, model_version, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result["request_id"], result["farm_id"],
                        result["risk_score"], result["risk_level"],
                        result["recommendation"], result["model_version"], source,
                    ),
                )
                await db.commit()
        except aiosqlite.Error as exc:
            raise self._unavailable() from exc

    async def save_batch(self, request_id: str, items: list[dict[str, Any]]) -> None:
        rows = [
            (
                request_id, item["farm_id"], item["risk_score"],
                item["risk_level"], item["recommendation"],
                item["model_version"], "batch",
            )
            for item in items
        ]
        try:
            async with self.connect() as db:
                await db.executemany(
                    """
                    INSERT INTO predictions (
                        request_id, farm_id, risk_score, risk_level,
                        recommendation, model_version, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                await db.commit()
        except aiosqlite.Error as exc:
            raise self._unavailable() from exc

    async def list_predictions(self, limit: int, risk_level: str | None = None) -> list[dict]:
        sql = """
            SELECT request_id, farm_id, risk_score, risk_level,
                   recommendation, model_version
            FROM predictions
        """
        params: tuple[Any, ...]
        if risk_level is None:
            sql += " ORDER BY id DESC LIMIT ?"
            params = (limit,)
        else:
            sql += " WHERE risk_level = ? ORDER BY id DESC LIMIT ?"
            params = (risk_level, limit)
        try:
            async with self.connect() as db:
                cursor = await db.execute(sql, params)
                rows = await cursor.fetchall()
                await cursor.close()
            return [dict(row) for row in rows]
        except aiosqlite.Error as exc:
            raise self._unavailable() from exc

    async def by_request_id(self, request_id: str) -> list[dict]:
        try:
            async with self.connect() as db:
                cursor = await db.execute(
                    """
                    SELECT request_id, farm_id, risk_score, risk_level,
                           recommendation, model_version
                    FROM predictions WHERE request_id = ? ORDER BY id
                    """,
                    (request_id,),
                )
                rows = await cursor.fetchall()
                await cursor.close()
            return [dict(row) for row in rows]
        except aiosqlite.Error as exc:
            raise self._unavailable() from exc

    async def weekday_page(
        self, weekday_number: str, page: int, per_page: int
    ) -> tuple[list[dict], int, int]:
        offset = (page - 1) * per_page
        try:
            async with self.connect() as db:
                cursor = await db.execute(
                    """
                    SELECT request_id, farm_id, risk_score, risk_level,
                           recommendation, model_version
                    FROM predictions
                    WHERE strftime('%w', created_at) = ?
                    ORDER BY id DESC LIMIT ? OFFSET ?
                    """,
                    (weekday_number, per_page, offset),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM predictions "
                    "WHERE strftime('%w', created_at) = ?",
                    (weekday_number,),
                )
                count_row = await cursor.fetchone()
                await cursor.close()
            total = int(count_row[0])
            pages = math.ceil(total / per_page) if total else 0
            return [dict(row) for row in rows], total, pages
        except aiosqlite.Error as exc:
            raise self._unavailable() from exc

    async def seed_weekdays(self, count: int) -> int:
        rows = []
        for index in range(count):
            score = round(min(0.1 + (index % 9) * 0.1, 1.0), 2)
            level = get_risk_level(score)
            rows.append(
                (
                    str(uuid.uuid4()), f"DEMO-{index + 1:03d}", score, level,
                    get_recommendation(level), self.model_version, "seed",
                    f"-{index % 21} days",
                )
            )
        try:
            async with self.connect() as db:
                await db.executemany(
                    """
                    INSERT INTO predictions (
                        request_id, farm_id, risk_score, risk_level,
                        recommendation, model_version, source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', ?))
                    """,
                    rows,
                )
                await db.commit()
            return count
        except aiosqlite.Error as exc:
            raise self._unavailable() from exc

    async def count(self) -> int:
        async with self.connect() as db:
            cursor = await db.execute("SELECT COUNT(*) FROM predictions")
            row = await cursor.fetchone()
            await cursor.close()
        return int(row[0])

    @staticmethod
    def _unavailable() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction storage is unavailable",
        )
