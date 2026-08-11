from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal, cast

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.projects.repository import Document, ProjectRepository
from app.modules.projects.schemas import (
    AgentModuleTokenUsage,
    DailyTokenUsage,
    ProjectCreate,
    ProjectDetail,
    ProjectList,
    ProjectSummary,
    ProjectTokenUsage,
    ProjectUpdate,
)

AgentModule = Literal["insight", "studio", "inspection"]
TOKEN_USAGE_MODULES: tuple[AgentModule, ...] = ("insight", "studio", "inspection")


def _summary(document: Document) -> ProjectSummary:
    return ProjectSummary(
        id=document["_id"],
        name=document["name"],
        description=document.get("description", ""),
        graph_revision=document.get("graph_revision", 0),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
    )


def _detail(document: Document) -> ProjectDetail:
    return ProjectDetail(
        **_summary(document).model_dump(),
        nodes=document.get("nodes", []),
        edges=document.get("edges", []),
    )


async def create_project(
    repository: ProjectRepository,
    principal: Principal,
    request: ProjectCreate,
) -> ProjectDetail:
    now = datetime.now(UTC)
    name = request.name.strip()
    document = {
        "_id": new_id("prj"),
        "owner_id": principal.user_id,
        "name": name,
        "name_key": name.casefold(),
        "description": request.description.strip(),
        "nodes": [],
        "edges": [],
        "graph_revision": 0,
        "created_at": now,
        "updated_at": now,
    }
    try:
        await repository.insert(document)
    except DuplicateKeyError as error:
        raise AppError(
            "project_name_conflict",
            "A project with this name already exists",
            status_code=409,
        ) from error
    return _detail(document)


async def list_projects(repository: ProjectRepository, principal: Principal) -> ProjectList:
    documents = await repository.list_for_owner(principal.user_id)
    return ProjectList(items=[_summary(document) for document in documents], total=len(documents))


async def get_project(
    repository: ProjectRepository,
    principal: Principal,
    project_id: str,
) -> ProjectDetail:
    document = await repository.get_for_owner(project_id, principal.user_id)
    if document is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return _detail(document)


async def update_project(
    repository: ProjectRepository,
    principal: Principal,
    project_id: str,
    request: ProjectUpdate,
) -> ProjectDetail:
    changes = request.model_dump(exclude_unset=True)
    update: Document = {"updated_at": datetime.now(UTC)}
    if "name" in changes:
        name = changes["name"].strip()
        update.update(name=name, name_key=name.casefold())
    if "description" in changes:
        update["description"] = changes["description"].strip()

    try:
        document = await repository.update_for_owner(project_id, principal.user_id, update)
    except DuplicateKeyError as error:
        raise AppError(
            "project_name_conflict",
            "A project with this name already exists",
            status_code=409,
        ) from error
    if document is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return _detail(document)


async def delete_project(
    repository: ProjectRepository,
    principal: Principal,
    project_id: str,
) -> None:
    deleted = await repository.delete_for_owner(project_id, principal.user_id)
    if not deleted:
        raise AppError("project_not_found", "Project was not found", status_code=404)


def _timezone_name(offset_minutes: int) -> str:
    sign = "+" if offset_minutes >= 0 else "-"
    hours, minutes = divmod(abs(offset_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


async def project_token_usage(
    repository: ProjectRepository,
    database: AsyncDatabase[dict[str, Any]],
    principal: Principal,
    project_id: str,
    *,
    days: int,
    timezone_offset_minutes: int,
) -> ProjectTokenUsage:
    if await repository.get_for_owner(project_id, principal.user_id) is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)

    offset = timedelta(minutes=timezone_offset_minutes)
    local_today = (datetime.now(UTC) + offset).date()
    today_start = datetime.combine(local_today, time.min, tzinfo=UTC) - offset
    tomorrow_start = today_start + timedelta(days=1)
    window_start = today_start - timedelta(days=days - 1)
    timezone = _timezone_name(timezone_offset_minutes)
    token_fields = {
        "input_tokens": {"$sum": {"$ifNull": ["$usage.input_tokens", 0]}},
        "output_tokens": {"$sum": {"$ifNull": ["$usage.output_tokens", 0]}},
    }
    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "owner_id": principal.user_id,
                "project_id": project_id,
                "completed_at": {"$gte": window_start, "$lt": tomorrow_start},
                "usage": {"$type": "object"},
            }
        },
        {
            "$facet": {
                "by_module": [
                    {"$match": {"completed_at": {"$gte": today_start}}},
                    {"$group": {"_id": {"$ifNull": ["$surface", "insight"]}, **token_fields}},
                ],
                "daily": [
                    {
                        "$group": {
                            "_id": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$completed_at",
                                    "timezone": timezone,
                                }
                            },
                            **token_fields,
                        }
                    }
                ],
            }
        },
    ]
    cursor = await database.agent_operations.aggregate(pipeline)
    results = await cursor.to_list(1)
    aggregate = results[0] if results else {"by_module": [], "daily": []}

    module_values = {
        str(item["_id"]): item
        for item in cast(list[dict[str, Any]], aggregate.get("by_module", []))
    }
    modules = [
        AgentModuleTokenUsage(
            module=module,
            input_tokens=int(module_values.get(module, {}).get("input_tokens", 0)),
            output_tokens=int(module_values.get(module, {}).get("output_tokens", 0)),
            total_tokens=(
                int(module_values.get(module, {}).get("input_tokens", 0))
                + int(module_values.get(module, {}).get("output_tokens", 0))
            ),
        )
        for module in TOKEN_USAGE_MODULES
    ]
    daily_values = {
        date.fromisoformat(str(item["_id"])): item
        for item in cast(list[dict[str, Any]], aggregate.get("daily", []))
    }
    daily: list[DailyTokenUsage] = []
    for index in range(days):
        day = local_today - timedelta(days=days - index - 1)
        value = daily_values.get(day, {})
        input_tokens = int(value.get("input_tokens", 0))
        output_tokens = int(value.get("output_tokens", 0))
        daily.append(
            DailyTokenUsage(
                date=day,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )
    return ProjectTokenUsage(
        today=local_today,
        today_total_tokens=sum(item.total_tokens for item in modules),
        by_module=modules,
        daily=daily,
    )
