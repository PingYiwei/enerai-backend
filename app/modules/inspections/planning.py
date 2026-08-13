from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.inspections.schemas import (
    DeviceInspectionManifest,
    InspectionGrade,
    InspectionPlanningManifest,
    InspectionRunCreate,
    InspectionTaskEdge,
    InspectionTaskGraph,
    InspectionTaskNode,
)
from app.modules.inspections.templates import template
from app.modules.projects.data import properties

Document = dict[str, Any]
GRADE_RANK: dict[str, int] = {"S": 4, "A": 3, "B": 2, "C": 1}


def node_grade(node: Document) -> InspectionGrade:
    raw_data = node.get("data")
    data: Document = raw_data if isinstance(raw_data, dict) else {}
    raw_inspection = data.get("inspection")
    inspection: Document = raw_inspection if isinstance(raw_inspection, dict) else {}
    raw = str(inspection.get("grade") or data.get("inspection_grade") or "B").upper()
    return raw if raw in GRADE_RANK else "B"  # type: ignore[return-value]


def inspection_enabled(node: Document) -> bool:
    raw_data = node.get("data")
    data: Document = raw_data if isinstance(raw_data, dict) else {}
    raw_inspection = data.get("inspection")
    inspection: Document = raw_inspection if isinstance(raw_inspection, dict) else {}
    return inspection.get("enabled") is not False


def node_label(node: Document) -> str:
    raw_data = node.get("data")
    data: Document = raw_data if isinstance(raw_data, dict) else {}
    return str(data.get("label") or data.get("name") or node.get("id") or "").strip()


async def locked_snapshot(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> Document:
    project = await database.projects.find_one({"_id": project_id, "owner_id": principal.user_id})
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    revision = int(project.get("graph_revision", 0))
    versions = getattr(database, "studio_graph_versions", None)
    version = (
        await versions.find_one(
            {"project_id": project_id, "owner_id": principal.user_id, "revision": revision}
        )
        if versions is not None and revision > 0
        else None
    )
    if version is None:
        return {
            **project,
            "nodes": project.get("nodes", []),
            "edges": project.get("edges", []),
            "graph_revision": revision,
        }
    return {
        **project,
        "nodes": version.get("nodes", []),
        "edges": version.get("edges", []),
        "graph_revision": revision,
        "snapshot_id": version.get("_id"),
    }


def _property_name(item: Document) -> str:
    return str(
        item.get("name") or item.get("property_id") or item.get("property_name") or ""
    ).strip()


def _property_device(item: Document) -> str:
    return str(item.get("device_id") or item.get("device_name") or "").strip()


async def _available_properties(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    devices: list[Document],
) -> tuple[dict[str, list[str]], Literal["available", "unavailable", "partial"], list[str]]:
    names = [node_label(node) for node in devices if node_label(node)]
    try:
        catalog = await properties(database, principal, project_id, device_ids=names)
    except AppError as error:
        return (
            {},
            "unavailable",
            [
                f"Operational data catalog is unavailable ({error.code}); all data-dependent "
                "dimensions require an inconclusive or skipped review."
            ],
        )
    by_device: dict[str, list[str]] = {}
    for raw in catalog.items:
        item = raw if isinstance(raw, dict) else {}
        device = _property_device(item)
        name = _property_name(item)
        if device and name and name not in by_device.setdefault(device, []):
            by_device[device].append(name)
    status: Literal["available", "unavailable", "partial"] = (
        "available" if all(by_device.get(name) for name in names) else "partial"
    )
    premises = []
    if status == "partial":
        premises.append(
            "The external catalog does not expose properties for every target device; planned "
            "unavailable properties must not be treated as operating faults by themselves."
        )
    return by_device, status, premises


def _declared_properties(node: Document) -> list[str]:
    raw_data = node.get("data")
    data: Document = raw_data if isinstance(raw_data, dict) else {}
    raw_sensors = data.get("sensors")
    sensors: list[Any] = raw_sensors if isinstance(raw_sensors, list) else []
    result: list[str] = []
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        name = str(sensor.get("name") or sensor.get("category") or "").strip()
        if name and name not in result:
            result.append(name)
    return result


def _scope_nodes(
    snapshot: Document, request: InspectionRunCreate, minimum: InspectionGrade
) -> list[Document]:
    return [
        node
        for node in snapshot.get("nodes", [])
        if isinstance(node, dict)
        and node.get("type") != "group"
        and inspection_enabled(node)
        and GRADE_RANK[node_grade(node)] >= GRADE_RANK[minimum]
    ]


def _related(snapshot: Document, node_id: str) -> list[str]:
    values: set[str] = set()
    for edge in snapshot.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source == node_id:
            values.add(target)
        elif target == node_id:
            values.add(source)
    return sorted(values)


def build_task_graph(
    snapshot: Document, manifests: list[DeviceInspectionManifest]
) -> InspectionTaskGraph:
    target_ids = {item.node_id for item in manifests}
    nodes = [
        InspectionTaskNode(
            id="stage:planning",
            kind="stage",
            title="Plan scope and data",
            status="succeeded",
            progress=1,
        ),
        InspectionTaskNode(
            id="stage:screening", kind="stage", title="Batch operational screening", status="ready"
        ),
        InspectionTaskNode(id="stage:review", kind="stage", title="Agent device review"),
    ]
    edges = [
        InspectionTaskEdge(
            id="flow:planning:screening",
            source="stage:planning",
            target="stage:screening",
            relation="flow",
        ),
        InspectionTaskEdge(
            id="flow:screening:review",
            source="stage:screening",
            target="stage:review",
            relation="flow",
        ),
    ]
    group_ids: set[str] = set()
    raw_by_id = {
        str(node.get("id")): node for node in snapshot.get("nodes", []) if isinstance(node, dict)
    }
    for manifest in manifests:
        parent_id = manifest.parent_id
        if parent_id and parent_id not in group_ids:
            group = raw_by_id.get(parent_id, {})
            nodes.append(
                InspectionTaskNode(
                    id=f"group:{parent_id}",
                    kind="group",
                    title=node_label(group) or parent_id,
                    reality_node_id=parent_id,
                    parent_id="stage:review",
                )
            )
            edges.append(
                InspectionTaskEdge(
                    id=f"contains:review:{parent_id}",
                    source="stage:review",
                    target=f"group:{parent_id}",
                    relation="contains",
                )
            )
            group_ids.add(parent_id)
        task_parent = f"group:{parent_id}" if parent_id else "stage:review"
        nodes.append(
            InspectionTaskNode(
                id=f"device:{manifest.node_id}",
                kind="device",
                title=manifest.node_label,
                status="pending",
                reality_node_id=manifest.node_id,
                parent_id=task_parent,
                grade=manifest.grade,
            )
        )
        edges.append(
            InspectionTaskEdge(
                id=f"contains:{task_parent}:{manifest.node_id}",
                source=task_parent,
                target=f"device:{manifest.node_id}",
                relation="contains",
            )
        )
    for edge in snapshot.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source in target_ids and target in target_ids:
            edges.append(
                InspectionTaskEdge(
                    id=f"feeds:{edge.get('id') or source + ':' + target}",
                    source=f"device:{source}",
                    target=f"device:{target}",
                    relation="feeds",
                )
            )
    nodes.extend(
        [
            InspectionTaskNode(id="stage:summary", kind="summary", title="Agent overall review"),
            InspectionTaskNode(
                id="stage:report", kind="report", title="Generate inspection report"
            ),
        ]
    )
    edges.extend(
        [
            InspectionTaskEdge(
                id="flow:review:summary",
                source="stage:review",
                target="stage:summary",
                relation="flow",
            ),
            InspectionTaskEdge(
                id="flow:summary:report",
                source="stage:summary",
                target="stage:report",
                relation="produces",
            ),
        ]
    )
    return InspectionTaskGraph(nodes=nodes, edges=edges)


def build_assignment_task_graph() -> InspectionTaskGraph:
    return InspectionTaskGraph(
        nodes=[
            InspectionTaskNode(
                id="stage:planning",
                kind="stage",
                title="Agent interpret assignment",
                status="ready",
            ),
            InspectionTaskNode(
                id="stage:execution",
                kind="stage",
                title="Execute Agent plan",
            ),
            InspectionTaskNode(
                id="stage:report",
                kind="report",
                title="Deliver assignment result",
            ),
        ],
        edges=[
            InspectionTaskEdge(
                id="flow:planning:execution",
                source="stage:planning",
                target="stage:execution",
                relation="flow",
            ),
            InspectionTaskEdge(
                id="flow:execution:report",
                source="stage:execution",
                target="stage:report",
                relation="produces",
            ),
        ],
    )


async def plan_inspection(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: InspectionRunCreate,
) -> tuple[Document, InspectionPlanningManifest, InspectionTaskGraph, str]:
    snapshot = await locked_snapshot(database, principal, project_id)
    selected_template = template(request.template_id or "full_inspection")
    minimum = request.minimum_grade or selected_template.default_minimum_grade
    if request.trigger == "assignment":
        now = datetime.now(UTC)
        manifest = InspectionPlanningManifest(
            reality_revision=int(snapshot.get("graph_revision", 0)),
            template_id="temporary_assignment",
            template_version=1,
            instruction=request.instruction.strip(),
            minimum_grade=minimum,
            window_start=now - timedelta(minutes=request.lookback_minutes),
            window_end=now,
            data_source_status="unavailable",
            premises=[
                "The temporary Assignment Agent determines the task scope and execution method.",
                (
                    "Only evidence returned by read-only project tools may be treated as "
                    "observed fact."
                ),
            ],
            devices=[],
        )
        return snapshot, manifest, build_assignment_task_graph(), "Temporary assignment"
    devices = _scope_nodes(snapshot, request, minimum)
    if not devices:
        raise AppError(
            "inspection_scope_empty",
            "No enabled Reality Model devices match this inspection scope",
            status_code=422,
        )
    available, source_status, source_premises = await _available_properties(
        database, principal, project_id, devices
    )
    now = datetime.now(UTC)
    manifests: list[DeviceInspectionManifest] = []
    for node in devices:
        node_id = str(node.get("id"))
        label = node_label(node)
        declared = _declared_properties(node)
        exposed = sorted(available.get(label, []))
        selected = sorted(set(declared) & set(exposed)) if declared else exposed
        skipped = sorted(set(declared) - set(exposed))
        premises: list[str] = []
        if skipped:
            premises.append(
                "These model-declared properties are unavailable and must be skipped: "
                + ", ".join(skipped)
            )
        if not exposed:
            premises.append(
                "No operational properties are exposed for this device; review structural and "
                "data-availability exceptions without inferring operating performance."
            )
        assessable = ["data_completeness", "sensor_mapping"]
        unavailable_dimensions: list[str] = []
        if selected:
            assessable.extend(
                [
                    "operating_condition",
                    "anomaly",
                    "efficiency",
                    "optimization",
                    "data_freshness",
                    "missingness",
                ]
            )
        else:
            unavailable_dimensions.extend(
                [
                    "operating_condition",
                    "efficiency",
                    "optimization",
                    "data_freshness",
                    "missingness",
                ]
            )
        manifests.append(
            DeviceInspectionManifest(
                node_id=node_id,
                node_label=label,
                node_type=str(node.get("type") or "equipment"),
                grade=node_grade(node),
                parent_id=str(node.get("parent_id")) if node.get("parent_id") else None,
                related_node_ids=_related(snapshot, node_id),
                declared_properties=declared,
                available_properties=exposed,
                selected_properties=selected,
                skipped_properties=skipped,
                premises=premises,
                assessable_dimensions=assessable,
                unavailable_dimensions=unavailable_dimensions,
            )
        )
    manifest = InspectionPlanningManifest(
        reality_revision=int(snapshot.get("graph_revision", 0)),
        template_id=selected_template.id,
        template_version=selected_template.version,
        instruction=request.instruction.strip(),
        minimum_grade=minimum,
        window_start=now - timedelta(minutes=request.lookback_minutes),
        window_end=now,
        data_source_status=source_status,
        premises=[
            (
                "All device conclusions must be Agent-reviewed, including skipped and "
                "inconclusive devices."
            ),
            (
                "Planned unavailable properties are evidence limitations, not proof of "
                "equipment failure."
            ),
            "Data completeness, freshness, and missingness are reportable abnormal dimensions.",
            *source_premises,
        ],
        devices=manifests,
    )
    return snapshot, manifest, build_task_graph(snapshot, manifests), selected_template.name
