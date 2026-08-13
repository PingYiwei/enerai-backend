from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.security import Principal
from app.modules.inspections.agent import InspectionCoordinator
from app.modules.inspections.schemas import InspectionRunCreate
from app.modules.inspections.service import create_run, due_schedules, next_schedule_run

Document = dict[str, Any]
logger = logging.getLogger(__name__)


class InspectionScheduler:
    def __init__(
        self,
        database: AsyncDatabase[Document],
        coordinator: InspectionCoordinator,
    ) -> None:
        self._database = database
        self._coordinator = coordinator
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
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
                await self._run_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduled inspection cycle failed")
            await asyncio.sleep(30)

    async def _run_due(self) -> None:
        now = datetime.now(UTC)
        for schedule in await due_schedules(self._database):
            claimed = await self._database.inspection_schedules.find_one_and_update(
                {"_id": schedule["_id"], "enabled": True, "next_run_at": schedule["next_run_at"]},
                {
                    "$set": {
                        "last_run_at": now,
                        "next_run_at": next_schedule_run(schedule, now),
                        "updated_at": now,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if claimed is None:
                continue
            principal = Principal(user_id=str(schedule["owner_id"]), username="")
            try:
                run = await create_run(
                    self._database,
                    principal,
                    str(schedule["project_id"]),
                    InspectionRunCreate(
                        trigger="manual",
                        template_id=schedule["template_id"],
                        minimum_grade=schedule.get("minimum_grade"),
                        lookback_minutes=int(schedule.get("lookback_minutes", 1_440)),
                    ),
                    trigger="schedule",
                    schedule_id=str(schedule["_id"]),
                )
                await self._coordinator.start(principal, run.id)
            except Exception:
                logger.exception("Failed to create scheduled inspection %s", schedule["_id"])
