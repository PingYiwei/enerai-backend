from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import DuplicateKeyError

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.inspections.schemas import (
    InspectionFinding,
    InspectionPolicy,
    InspectionPolicyUpdate,
    InspectionRun,
    InspectionRunList,
)

Document = dict[str, Any]


async def _owned_project(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> Document:
    project = await database.projects.find_one({"_id": project_id, "owner_id": principal.user_id})
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return project


async def get_policy(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> InspectionPolicy:
    await _owned_project(database, principal, project_id)
    document = await database.inspection_policies.find_one({"project_id": project_id})
    if document is None:
        return InspectionPolicy(project_id=project_id, updated_at=datetime.now(UTC))
    return InspectionPolicy.model_validate(document)


async def save_policy(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: InspectionPolicyUpdate,
) -> InspectionPolicy:
    await _owned_project(database, principal, project_id)
    now = datetime.now(UTC)
    document = {
        "_id": project_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        **request.model_dump(mode="json"),
        "updated_at": now,
    }
    await database.inspection_policies.replace_one(
        {"project_id": project_id}, document, upsert=True
    )
    return InspectionPolicy.model_validate(document)


def inspect_graph(project: Document) -> list[InspectionFinding]:
    nodes = project.get("nodes", [])
    edges = project.get("edges", [])
    findings: list[InspectionFinding] = []
    if not nodes:
        return [
            InspectionFinding(
                code="graph_empty",
                severity="warning",
                title="Studio graph is empty",
                detail="Add equipment and connections before operational inspection.",
            )
        ]

    node_ids = {str(node["id"]) for node in nodes}
    connected = {
        str(endpoint)
        for edge in edges
        for endpoint in (edge.get("source"), edge.get("target"))
        if endpoint is not None
    }
    isolated = sorted(node_ids - connected)
    if isolated:
        findings.append(
            InspectionFinding(
                code="isolated_equipment",
                severity="warning",
                title="Equipment is disconnected",
                detail=f"{len(isolated)} graph nodes have no connection.",
                node_ids=isolated,
            )
        )
    sensors = [node for node in nodes if node.get("type") == "sensor"]
    unmapped = sorted(
        str(node["id"])
        for node in sensors
        if not isinstance(node.get("data"), dict) or not node["data"].get("property")
    )
    if unmapped:
        findings.append(
            InspectionFinding(
                code="sensor_property_missing",
                severity="critical",
                title="Sensor property is not mapped",
                detail=f"{len(unmapped)} sensors cannot resolve an operational property.",
                node_ids=unmapped,
            )
        )
    if not findings:
        findings.append(
            InspectionFinding(
                code="graph_integrity_ok",
                severity="info",
                title="Graph integrity checks passed",
                detail="All current structural checks completed without findings.",
            )
        )
    return findings


async def create_run(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    trigger: Literal["manual", "schedule"] = "manual",
) -> InspectionRun:
    project = await _owned_project(database, principal, project_id)
    now = datetime.now(UTC)
    document = {
        "_id": new_id("isr"),
        "project_id": project_id,
        "owner_id": principal.user_id,
        "status": "completed",
        "trigger": trigger,
        "graph_revision": int(project.get("graph_revision", 0)),
        "findings": [item.model_dump(mode="json") for item in inspect_graph(project)],
        "started_at": now,
        "completed_at": now,
    }
    await database.inspection_runs.insert_one(document)
    return InspectionRun.model_validate(document)


async def run_due_policies(database: AsyncDatabase[Document]) -> int:
    now = datetime.now(UTC)
    policies = await database.inspection_policies.find({"enabled": True}).to_list(None)
    created = 0
    for policy in policies:
        interval_seconds = int(policy["interval_minutes"]) * 60
        slot = int(now.timestamp()) // interval_seconds
        project = await database.projects.find_one(
            {"_id": policy["project_id"], "owner_id": policy["owner_id"]}
        )
        if project is None:
            continue
        document = {
            "_id": new_id("isr"),
            "project_id": project["_id"],
            "owner_id": policy["owner_id"],
            "status": "completed",
            "trigger": "schedule",
            "schedule_slot": slot,
            "graph_revision": int(project.get("graph_revision", 0)),
            "findings": [item.model_dump(mode="json") for item in inspect_graph(project)],
            "started_at": now,
            "completed_at": now,
        }
        try:
            await database.inspection_runs.insert_one(document)
            created += 1
        except DuplicateKeyError:
            continue
    return created


async def list_runs(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> InspectionRunList:
    await _owned_project(database, principal, project_id)
    documents = await (
        database.inspection_runs.find({"project_id": project_id, "owner_id": principal.user_id})
        .sort("started_at", DESCENDING)
        .to_list(None)
    )
    return InspectionRunList(
        items=[InspectionRun.model_validate(document) for document in documents],
        total=len(documents),
    )
