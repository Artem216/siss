"""Очередь, backpressure и динамическое формирование batch."""

import asyncio
import logging
import time
from dataclasses import dataclass

from .config import Settings
from .model import ModelEngine
from .schemas import FarmRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future[float]


class DynamicBatcher:
    def __init__(self, settings: Settings, engine: ModelEngine) -> None:
        self.settings = settings
        self.engine = engine
        self.queue: asyncio.Queue[InferenceTask] = asyncio.Queue(
            maxsize=settings.queue_max_size
        )
        self.worker: asyncio.Task | None = None

    def start(self) -> None:
        self.worker = asyncio.create_task(self._run(), name="dynamic-batch-worker")

    async def stop(self) -> None:
        if self.worker is None:
            return
        self.worker.cancel()
        try:
            await self.worker
        except asyncio.CancelledError:
            pass

    async def submit(self, data: FarmRequest, request_id: str) -> float:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[float] = loop.create_future()
        self.queue.put_nowait(InferenceTask(data, request_id, future))
        try:
            return await asyncio.wait_for(
                future, timeout=self.settings.inference_timeout_seconds
            )
        except asyncio.TimeoutError:
            if not future.done():
                future.cancel()
            raise

    async def _run(self) -> None:
        logger.info("Dynamic batch worker started")
        while True:
            first = await self.queue.get()
            if first.future.cancelled():
                self.queue.task_done()
                continue
            batch = [first]
            deadline = time.perf_counter() + self.settings.max_batch_wait_ms / 1000
            while len(batch) < self.settings.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    task = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if task.future.cancelled():
                    self.queue.task_done()
                    continue
                batch.append(task)
            logger.info(
                "Dynamic batch formed | size=%s | queue_size=%s",
                len(batch), self.queue.qsize(),
            )
            try:
                scores = await asyncio.wait_for(
                    self.engine.predict_batch([task.data for task in batch]),
                    timeout=self.settings.inference_timeout_seconds,
                )
                for task, score in zip(batch, scores):
                    if not task.future.done():
                        task.future.set_result(score)
            except Exception as exc:
                logger.exception("Dynamic batch inference failed")
                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

