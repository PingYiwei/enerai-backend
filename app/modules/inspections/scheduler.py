from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.modules.inspections.service import run_due_policies

Document = dict[str, Any]
logger = logging.getLogger(__name__)


class InspectionScheduler:
    def __init__(self, database: AsyncDatabase[Document]) -> None:
        self._database = database
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="inspection-scheduler")

    async def close(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await run_due_policies(self._database)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduled inspection cycle failed")
            await asyncio.sleep(30)
