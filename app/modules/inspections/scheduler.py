from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.modules.inspections.service import run_due_policies

Document = dict[str, Any]


class InspectionScheduler:
    def __init__(self, database: AsyncDatabase[Document]) -> None:
        self._database = database
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
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
            await run_due_policies(self._database)
            await asyncio.sleep(30)
