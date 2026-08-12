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
    CheckId,
    InspectionFinding,
    InspectionPolicy,
    InspectionPolicyUpdate,
    InspectionRun,
    InspectionRunList,
    default_checks,
)

Document = dict[str, Any]


def _inspection_run(document: Document) -> InspectionRun:
    payload = {**document, "id": str(document.get("id") or document.get("_id") or "")}
    return InspectionRun.model_validate(payload)


def _policy_checks(document: Document | None) -> list[CheckId]:
    if document is None:
        return default_checks()
    return InspectionPolicyUpdate.model_validate(document).checks


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


def inspect_graph(
    project: Document, checks: list[CheckId] | None = None
) -> list[InspectionFinding]:
    selected_checks = set(checks or default_checks())
    nodes = [node for node in project.get("nodes", []) if isinstance(node, dict)]
    edges = [edge for edge in project.get("edges", []) if isinstance(edge, dict)]
    findings: list[InspectionFinding] = []

    if "graph_integrity" in selected_checks and not nodes:
        findings.append(
            InspectionFinding(
                code="graph_empty",
                severity="warning",
                title="Studio graph is empty",
                detail="Add equipment and connections before operational inspection.",
            )
        )

    if "graph_integrity" in selected_checks and nodes:
        node_ids = {
            str(node["id"])
            for node in nodes
            if node.get("id") is not None and node.get("type") != "group"
        }
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

    if "sensor_coverage" in selected_checks:
        incomplete_sensor_nodes: set[str] = set()
        for node in nodes:
            node_id = str(node.get("id") or "")
            raw_data = node.get("data")
            data: Document = raw_data if isinstance(raw_data, dict) else {}
            if node.get("type") == "sensor" and not data.get("property"):
                incomplete_sensor_nodes.add(node_id)
            embedded_sensors = data.get("sensors", [])
            if not isinstance(embedded_sensors, list):
                incomplete_sensor_nodes.add(node_id)
                continue
            if any(
                not isinstance(sensor, dict)
                or not str(sensor.get("name") or "").strip()
                or not str(sensor.get("category") or "").strip()
                for sensor in embedded_sensors
            ):
                incomplete_sensor_nodes.add(node_id)
        incomplete_sensor_nodes.discard("")
        if incomplete_sensor_nodes:
            incomplete_node_ids = sorted(incomplete_sensor_nodes)
            findings.append(
                InspectionFinding(
                    code="sensor_property_missing",
                    severity="critical",
                    title="Sensor definition is incomplete",
                    detail=(
                        f"{len(incomplete_node_ids)} graph nodes contain sensors "
                        "without a name or category."
                    ),
                    node_ids=incomplete_node_ids,
                )
            )

    if "data_freshness" in selected_checks:
        findings.append(
            InspectionFinding(
                code="data_freshness_unavailable",
                severity="warning",
                title="Data freshness check is unavailable",
                detail="Operational data freshness is not connected to the inspection runtime yet.",
            )
        )

    if not findings:
        findings.append(
            InspectionFinding(
                code="graph_integrity_ok",
                severity="info",
                title="Configured inspection checks passed",
                detail="All configured structural checks completed without findings.",
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
    policy = await database.inspection_policies.find_one(
        {"project_id": project_id, "owner_id": principal.user_id}
    )
    checks = _policy_checks(policy)
    now = datetime.now(UTC)
    document = {
        "_id": new_id("isr"),
        "project_id": project_id,
        "owner_id": principal.user_id,
        "status": "completed",
        "trigger": trigger,
        "checks": checks,
        "graph_revision": int(project.get("graph_revision", 0)),
        "findings": [item.model_dump(mode="json") for item in inspect_graph(project, checks)],
        "started_at": now,
        "completed_at": now,
    }
    await database.inspection_runs.insert_one(document)
    return _inspection_run(document)


async def run_due_policies(database: AsyncDatabase[Document]) -> int:
    now = datetime.now(UTC)
    policies = await database.inspection_policies.find({"enabled": True}).to_list(None)
    created = 0
    for policy in policies:
        checks = _policy_checks(policy)
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
            "checks": checks,
            "schedule_slot": slot,
            "graph_revision": int(project.get("graph_revision", 0)),
            "findings": [item.model_dump(mode="json") for item in inspect_graph(project, checks)],
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
        items=[_inspection_run(document) for document in documents],
        total=len(documents),
    )
