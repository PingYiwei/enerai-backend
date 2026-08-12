from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import APIRouter, Header, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.api.dependencies import CurrentPrincipal, Database
from app.modules.inspections.agent import InspectionCoordinator
from app.modules.inspections.schemas import (
    InspectionPolicy,
    InspectionPolicyUpdate,
    InspectionReport,
    InspectionRun,
    InspectionRunCreate,
    InspectionRunList,
    InspectionSchedule,
    InspectionScheduleCreate,
    InspectionScheduleList,
    InspectionScheduleUpdate,
    InspectionTemplateList,
)
from app.modules.inspections.service import (
    create_run,
    create_schedule,
    delete_schedule,
    get_policy,
    get_run,
    list_runs,
    list_schedules,
    save_policy,
    update_schedule,
)
from app.modules.inspections.templates import TEMPLATES

router = APIRouter()
TERMINAL_STATES = {"completed", "partial", "failed", "cancelled"}


def coordinator(request: Request) -> InspectionCoordinator:
    return cast(InspectionCoordinator, request.app.state.inspection_coordinator)


def event_data_json(data: object) -> str:
    return json.dumps(
        jsonable_encoder(data),
        ensure_ascii=False,
        separators=(",", ":"),
    )


@router.get("/templates", response_model=InspectionTemplateList)
async def templates() -> InspectionTemplateList:
    return InspectionTemplateList(items=list(TEMPLATES.values()))


@router.get("/policy", response_model=InspectionPolicy)
async def policy(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionPolicy:
    return await get_policy(database, principal, project_id)


@router.put("/policy", response_model=InspectionPolicy)
async def update_policy(
    project_id: str,
    body: InspectionPolicyUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> InspectionPolicy:
    return await save_policy(database, principal, project_id, body)


@router.post("/schedules", response_model=InspectionSchedule, status_code=status.HTTP_201_CREATED)
async def add_schedule(
    project_id: str,
    body: InspectionScheduleCreate,
    database: Database,
    principal: CurrentPrincipal,
) -> InspectionSchedule:
    return await create_schedule(database, principal, project_id, body)


@router.get("/schedules", response_model=InspectionScheduleList)
async def schedules(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionScheduleList:
    return await list_schedules(database, principal, project_id)


@router.patch("/schedules/{schedule_id}", response_model=InspectionSchedule)
async def patch_schedule(
    schedule_id: str,
    body: InspectionScheduleUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> InspectionSchedule:
    return await update_schedule(database, principal, schedule_id, body)


@router.delete("/schedules/{schedule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_schedule(
    schedule_id: str, database: Database, principal: CurrentPrincipal
) -> Response:
    await delete_schedule(database, principal, schedule_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/runs", response_model=InspectionRun, status_code=status.HTTP_201_CREATED)
async def plan_run(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
    body: InspectionRunCreate | None = None,
) -> InspectionRun:
    return await create_run(database, principal, project_id, body)


@router.post(
    "/runs/{run_id}/start", response_model=InspectionRun, status_code=status.HTTP_202_ACCEPTED
)
async def start_run(
    project_id: str,
    run_id: str,
    request: Request,
    database: Database,
    principal: CurrentPrincipal,
) -> InspectionRun:
    await get_run(database, principal, run_id, project_id)
    await coordinator(request).start(principal, run_id)
    return await get_run(database, principal, run_id, project_id)


@router.post("/runs/{run_id}/cancel", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_run(
    project_id: str,
    run_id: str,
    request: Request,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    await get_run(database, principal, run_id, project_id)
    await coordinator(request).cancel(principal, run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs", response_model=InspectionRunList)
async def runs(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionRunList:
    return await list_runs(database, principal, project_id)


@router.get("/runs/{run_id}", response_model=InspectionRun)
async def run(
    project_id: str, run_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionRun:
    return await get_run(database, principal, run_id, project_id)


@router.get("/runs/{run_id}/report", response_model=InspectionReport)
async def report(
    project_id: str, run_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionReport:
    inspection = await get_run(database, principal, run_id, project_id)
    if inspection.report is None:
        from app.core.errors import AppError

        raise AppError(
            "inspection_report_not_ready", "Inspection report is not ready", status_code=409
        )
    return inspection.report


@router.get("/runs/{run_id}/events")
async def stream_events(
    project_id: str,
    run_id: str,
    database: Database,
    principal: CurrentPrincipal,
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    cursor = max(after, int(last_event_id or 0))
    await get_run(database, principal, run_id, project_id)

    async def events() -> AsyncIterator[str]:
        nonlocal cursor
        idle_polls = 0
        while True:
            document = await database.inspection_runs.find_one(
                {
                    "_id": run_id,
                    "project_id": project_id,
                    "owner_id": principal.user_id,
                },
                {"status": 1},
            )
            if document is None:
                return
            rows = await (
                database.inspection_events.find(
                    {
                        "run_id": run_id,
                        "project_id": project_id,
                        "owner_id": principal.user_id,
                        "seq": {"$gt": cursor},
                    }
                )
                .sort("seq", 1)
                .to_list(None)
            )
            for row in rows:
                cursor = int(row["seq"])
                data = event_data_json(row["data"])
                yield f"id: {cursor}\nevent: {row['type']}\ndata: {data}\n\n"
            if document.get("status") in TERMINAL_STATES and not rows:
                return
            idle_polls += 1
            if idle_polls % 60 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
