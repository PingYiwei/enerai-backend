from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.inspections.planning import plan_inspection
from app.modules.inspections.schemas import (
    CheckId,
    InspectionFinding,
    InspectionPolicy,
    InspectionPolicyUpdate,
    InspectionRun,
    InspectionRunCreate,
    InspectionRunList,
    InspectionSchedule,
    InspectionScheduleCreate,
    InspectionScheduleList,
    InspectionScheduleUpdate,
    default_checks,
)
from app.modules.inspections.templates import template

Document = dict[str, Any]


async def owned_project(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> Document:
    project = await database.projects.find_one({"_id": project_id, "owner_id": principal.user_id})
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return project


def _inspection_run(document: Document) -> InspectionRun:
    now = datetime.now(UTC)
    payload = {
        **document,
        "id": str(document.get("id") or document.get("_id") or ""),
        "started_at": document.get("started_at") or document.get("created_at") or now,
        "completed_at": document.get("completed_at"),
    }
    return InspectionRun.model_validate(payload)


def _schedule(document: Document) -> InspectionSchedule:
    return InspectionSchedule.model_validate({**document, "id": str(document["_id"])})


async def append_event(
    database: AsyncDatabase[Document], run_id: str, event_type: str, data: Document
) -> None:
    run = await database.inspection_runs.find_one_and_update(
        {"_id": run_id},
        {"$inc": {"event_sequence": 1}, "$set": {"updated_at": datetime.now(UTC)}},
        return_document=ReturnDocument.AFTER,
    )
    if run is None:
        raise AppError("inspection_run_not_found", "Inspection run was not found", status_code=404)
    await database.inspection_events.insert_one(
        {
            "_id": f"{run_id}:{run['event_sequence']}",
            "run_id": run_id,
            "project_id": run["project_id"],
            "owner_id": run["owner_id"],
            "seq": int(run["event_sequence"]),
            "type": event_type,
            "data": data,
            "created_at": datetime.now(UTC),
        }
    )


async def create_run(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: InspectionRunCreate | None = None,
    *,
    trigger: str | None = None,
    schedule_id: str | None = None,
) -> InspectionRun:
    body = request or InspectionRunCreate()
    if trigger == "schedule":
        body = body.model_copy(update={"trigger": "manual"})
    snapshot, planning, task_graph, template_name = await plan_inspection(
        database, principal, project_id, body
    )
    now = datetime.now(UTC)
    run_id = new_id("isr")
    document: Document = {
        "_id": run_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        "status": "queued" if trigger == "schedule" else "ready",
        "trigger": trigger or body.trigger,
        "schedule_id": schedule_id,
        "template_id": planning.template_id,
        "template_name": template_name,
        "template_version": planning.template_version,
        "minimum_grade": planning.minimum_grade,
        "instruction": body.instruction.strip(),
        "graph_revision": planning.reality_revision,
        "snapshot": {
            "revision": planning.reality_revision,
            "nodes": snapshot.get("nodes", []),
            "edges": snapshot.get("edges", []),
        },
        "planning_manifest": planning.model_dump(mode="json"),
        "task_graph": task_graph.model_dump(mode="json"),
        "node_results": [],
        "findings": [],
        "overall_conclusion": None,
        "report": None,
        "requested_provider": body.provider,
        "requested_api_style": body.api_style,
        "requested_model": body.model,
        "provider": None,
        "model": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        },
        "progress": 0,
        "event_sequence": 0,
        "error": None,
        "created_at": now,
        "started_at": now,
        "completed_at": None,
        "updated_at": now,
        "checks": default_checks(),
    }
    await database.inspection_runs.insert_one(document)
    events = getattr(database, "inspection_events", None)
    if events is not None:
        await append_event(
            database,
            run_id,
            "graph_ready",
            {
                "task_graph": document["task_graph"],
                "planning_manifest": document["planning_manifest"],
            },
        )
    return _inspection_run(document)


async def get_run(
    database: AsyncDatabase[Document],
    principal: Principal,
    run_id: str,
    project_id: str | None = None,
) -> InspectionRun:
    query: Document = {"_id": run_id, "owner_id": principal.user_id}
    if project_id is not None:
        query["project_id"] = project_id
    document = await database.inspection_runs.find_one(query)
    if document is None:
        raise AppError("inspection_run_not_found", "Inspection run was not found", status_code=404)
    return _inspection_run(document)


async def list_runs(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> InspectionRunList:
    await owned_project(database, principal, project_id)
    documents = await (
        database.inspection_runs.find({"project_id": project_id, "owner_id": principal.user_id})
        .sort("created_at", DESCENDING)
        .to_list(None)
    )
    return InspectionRunList(
        items=[_inspection_run(item) for item in documents], total=len(documents)
    )


async def get_policy(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> InspectionPolicy:
    await owned_project(database, principal, project_id)
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
    await owned_project(database, principal, project_id)
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


async def create_schedule(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    body: InspectionScheduleCreate,
) -> InspectionSchedule:
    await owned_project(database, principal, project_id)
    selected = template(body.template_id)
    now = datetime.now(UTC)
    document = {
        "_id": new_id("iss"),
        "project_id": project_id,
        "owner_id": principal.user_id,
        **body.model_dump(mode="json"),
        "minimum_grade": body.minimum_grade or selected.default_minimum_grade,
        "next_run_at": now + timedelta(minutes=body.interval_minutes),
        "last_run_at": None,
        "created_at": now,
        "updated_at": now,
    }
    await database.inspection_schedules.insert_one(document)
    return _schedule(document)


async def list_schedules(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> InspectionScheduleList:
    await owned_project(database, principal, project_id)
    documents = await (
        database.inspection_schedules.find(
            {"project_id": project_id, "owner_id": principal.user_id}
        )
        .sort("created_at", DESCENDING)
        .to_list(None)
    )
    return InspectionScheduleList(
        items=[_schedule(item) for item in documents], total=len(documents)
    )


async def update_schedule(
    database: AsyncDatabase[Document],
    principal: Principal,
    schedule_id: str,
    body: InspectionScheduleUpdate,
) -> InspectionSchedule:
    current = await database.inspection_schedules.find_one(
        {"_id": schedule_id, "owner_id": principal.user_id}
    )
    if current is None:
        raise AppError("inspection_schedule_not_found", "Schedule was not found", status_code=404)
    values = body.model_dump(exclude_none=True, mode="json")
    if "template_id" in values and "minimum_grade" not in values:
        values["minimum_grade"] = template(values["template_id"]).default_minimum_grade
    if "interval_minutes" in values:
        values["next_run_at"] = datetime.now(UTC) + timedelta(minutes=values["interval_minutes"])
    values["updated_at"] = datetime.now(UTC)
    document = await database.inspection_schedules.find_one_and_update(
        {"_id": schedule_id, "owner_id": principal.user_id},
        {"$set": values},
        return_document=ReturnDocument.AFTER,
    )
    return _schedule(document or current)


async def delete_schedule(
    database: AsyncDatabase[Document], principal: Principal, schedule_id: str
) -> None:
    result = await database.inspection_schedules.delete_one(
        {"_id": schedule_id, "owner_id": principal.user_id}
    )
    if not result.deleted_count:
        raise AppError("inspection_schedule_not_found", "Schedule was not found", status_code=404)


async def due_schedules(database: AsyncDatabase[Document]) -> list[Document]:
    return await (
        database.inspection_schedules.find(
            {"enabled": True, "next_run_at": {"$lte": datetime.now(UTC)}}
        )
        .sort("next_run_at", ASCENDING)
        .to_list(None)
    )


# Legacy structural checks remain available for historical callers and focused unit tests.
def inspect_graph(
    project: Document, checks: list[CheckId] | None = None
) -> list[InspectionFinding]:
    selected = set(checks or default_checks())
    nodes = [item for item in project.get("nodes", []) if isinstance(item, dict)]
    edges = [item for item in project.get("edges", []) if isinstance(item, dict)]
    findings: list[InspectionFinding] = []
    if "graph_integrity" in selected and not nodes:
        findings.append(
            InspectionFinding(
                code="graph_empty",
                severity="warning",
                category="topology",
                title="Studio graph is empty",
                detail="Add equipment and connections before operational inspection.",
            )
        )
    if "graph_integrity" in selected and nodes:
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
                    category="topology",
                    title="Equipment is disconnected",
                    detail=f"{len(isolated)} graph nodes have no connection.",
                    node_ids=isolated,
                )
            )
    if "sensor_coverage" in selected:
        incomplete: set[str] = set()
        for node in nodes:
            raw_data = node.get("data")
            data: Document = raw_data if isinstance(raw_data, dict) else {}
            raw_sensors = data.get("sensors")
            sensors: list[Any] = raw_sensors if isinstance(raw_sensors, list) else []
            if node.get("type") == "sensor" and not data.get("property"):
                incomplete.add(str(node.get("id") or ""))
            if not isinstance(sensors, list) or any(
                not isinstance(sensor, dict)
                or not str(sensor.get("name") or "").strip()
                or not str(sensor.get("category") or "").strip()
                for sensor in sensors
            ):
                incomplete.add(str(node.get("id") or ""))
        incomplete.discard("")
        if incomplete:
            findings.append(
                InspectionFinding(
                    code="sensor_property_missing",
                    severity="critical",
                    category="data_completeness",
                    title="Sensor definition is incomplete",
                    detail=f"{len(incomplete)} graph nodes contain incomplete sensors.",
                    node_ids=sorted(incomplete),
                )
            )
    if "data_freshness" in selected:
        findings.append(
            InspectionFinding(
                code="data_freshness_unavailable",
                severity="warning",
                category="data_freshness",
                title="Data freshness check is unavailable",
                detail="Operational data freshness is unavailable in the legacy checker.",
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
