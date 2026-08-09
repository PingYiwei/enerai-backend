from __future__ import annotations

from datetime import UTC, datetime

from pymongo.errors import DuplicateKeyError

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.projects.repository import Document, ProjectRepository
from app.modules.projects.schemas import (
    ProjectCreate,
    ProjectDetail,
    ProjectList,
    ProjectSummary,
    ProjectUpdate,
)


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
