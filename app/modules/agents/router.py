from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, cast
from urllib.parse import quote

from fastapi import APIRouter, File, Header, Query, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import CurrentPrincipal, Database
from app.modules.agents.artifacts import list_artifacts, read_artifact
from app.modules.agents.attachments import (
    create_attachment,
    delete_draft_attachment,
    read_attachment,
)
from app.modules.agents.context import context_options
from app.modules.agents.repository import MongoAgentRepository
from app.modules.agents.schemas import (
    ArtifactList,
    AttachmentSummary,
    ContextOptions,
    RunAccepted,
    RunCreate,
    RunStatus,
    SessionCreate,
    SessionList,
    SessionSnapshot,
    SessionSummary,
    SessionUpdate,
)
from app.modules.agents.service import AgentRunCoordinator

router = APIRouter()
TERMINAL_RUN_STATES = {"completed", "failed", "cancelled"}


@router.get("/projects/{project_id}/insight/context-options", response_model=ContextOptions)
async def get_context_options(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> ContextOptions:
    return await context_options(database, principal, project_id)


@router.post(
    "/projects/{project_id}/insight/attachments",
    response_model=AttachmentSummary,
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(
    project_id: str,
    file: Annotated[UploadFile, File()],
    database: Database,
    principal: CurrentPrincipal,
) -> AttachmentSummary:
    return await create_attachment(database, principal, project_id, file)


@router.get("/attachments/{attachment_id}/content")
async def attachment_content(
    attachment_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    attachment, content = await read_attachment(database, principal, attachment_id)
    return Response(content, media_type=attachment.media_type)


@router.delete("/attachments/{attachment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_attachment(
    attachment_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    await delete_draft_attachment(database, principal, attachment_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/sessions/{session_id}/artifacts", response_model=ArtifactList)
async def session_artifacts(
    session_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> ArtifactList:
    return await list_artifacts(database, principal, session_id)


@router.get("/artifacts/{artifact_id}/content")
async def artifact_content(
    artifact_id: str,
    database: Database,
    principal: CurrentPrincipal,
    download: Annotated[bool, Query()] = False,
) -> Response:
    artifact, content = await read_artifact(database, principal, artifact_id)
    disposition = "attachment" if download else "inline"
    return Response(
        content,
        media_type=artifact.media_type,
        headers={
            "Content-Disposition": (
                f"{disposition}; filename=artifact; filename*=UTF-8''{quote(artifact.file_name)}"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


def repository(database: Database) -> MongoAgentRepository:
    return MongoAgentRepository(database)


def coordinator(request: Request) -> AgentRunCoordinator:
    return cast(AgentRunCoordinator, request.app.state.agent_coordinator)


@router.post(
    "/projects/{project_id}/insight/sessions",
    response_model=SessionSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    project_id: str,
    body: SessionCreate,
    database: Database,
    principal: CurrentPrincipal,
) -> SessionSummary:
    return await repository(database).create_session(principal, project_id, body)


@router.get("/projects/{project_id}/insight/sessions", response_model=SessionList)
async def list_sessions(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> SessionList:
    return await repository(database).list_sessions(principal, project_id)


@router.post(
    "/projects/{project_id}/studio/sessions",
    response_model=SessionSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_studio_session(
    project_id: str,
    body: SessionCreate,
    database: Database,
    principal: CurrentPrincipal,
) -> SessionSummary:
    return await repository(database).create_session(principal, project_id, body, surface="studio")


@router.get("/projects/{project_id}/studio/sessions", response_model=SessionList)
async def list_studio_sessions(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> SessionList:
    return await repository(database).list_sessions(principal, project_id, surface="studio")


@router.get("/sessions/{session_id}", response_model=SessionSnapshot)
async def get_snapshot(
    session_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> SessionSnapshot:
    return await repository(database).snapshot(principal, session_id)


@router.patch("/sessions/{session_id}", response_model=SessionSummary)
async def update_session(
    session_id: str,
    body: SessionUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> SessionSummary:
    return await repository(database).update_session(principal, session_id, body)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> None:
    await repository(database).delete_session(principal, session_id)


@router.post("/sessions/{session_id}/runs", response_model=RunAccepted, status_code=202)
async def create_run(
    session_id: str,
    body: RunCreate,
    request: Request,
    database: Database,
    principal: CurrentPrincipal,
) -> RunAccepted:
    return await coordinator(request).start(
        repository(database),
        principal,
        session_id,
        body,
    )


@router.get("/runs/{run_id}", response_model=RunStatus)
async def get_run(
    run_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> RunStatus:
    return await repository(database).get_run(principal, run_id)


@router.post("/runs/{run_id}/abort", status_code=204)
async def abort_run(
    run_id: str,
    request: Request,
    database: Database,
    principal: CurrentPrincipal,
) -> None:
    await coordinator(request).abort(repository(database), principal, run_id)


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    database: Database,
    principal: CurrentPrincipal,
    after: Annotated[int, Query(ge=0)] = 0,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    cursor = max(after, int(last_event_id or 0))
    agent_repository = repository(database)
    await agent_repository.get_run(principal, run_id)

    async def events() -> AsyncIterator[str]:
        nonlocal cursor
        idle_polls = 0
        while True:
            run, documents = await agent_repository.list_events(principal, run_id, cursor)
            for document in documents:
                cursor = int(document["seq"])
                data = json.dumps(document["data"], ensure_ascii=False, separators=(",", ":"))
                yield f"id: {cursor}\nevent: {document['type']}\ndata: {data}\n\n"
            if run.status in TERMINAL_RUN_STATES and not documents:
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
