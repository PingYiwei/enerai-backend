from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.studio.schemas import (
    CatalogItem,
    CategoryGroup,
    CategoryOption,
    StudioCatalog,
    StudioCategories,
    StudioGraph,
    StudioGraphUpdate,
)

Document = dict[str, Any]

CATALOG = StudioCatalog(
    items=[
        CatalogItem(
            type="chiller",
            label="Chiller",
            category="Cooling",
            description="Produces chilled water for the cooling loop.",
        ),
        CatalogItem(
            type="cooling_tower",
            label="Cooling tower",
            category="Cooling",
            description="Rejects condenser-loop heat to ambient air.",
        ),
        CatalogItem(
            type="pump",
            label="Pump",
            category="Hydronics",
            description="Moves water through a hydronic circuit.",
        ),
        CatalogItem(
            type="heat_exchanger",
            label="Heat exchanger",
            category="Hydronics",
            description="Transfers heat between isolated circuits.",
        ),
        CatalogItem(
            type="pipe",
            label="Pipe",
            category="Hydronics",
            description="Connects equipment within a hydronic circuit.",
        ),
        CatalogItem(
            type="valve",
            label="Valve",
            category="Hydronics",
            description="Regulates or isolates flow through a circuit.",
        ),
        CatalogItem(
            type="group",
            label="Group",
            category="Structure",
            description="Groups equipment without changing its coordinates.",
        ),
    ]
)


def _document_text(document: Document, *keys: str) -> str:
    for key in keys:
        value = document.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _normalize_categories(documents: list[Document]) -> list[CategoryGroup]:
    grouped: dict[str, dict[str, CategoryOption]] = {}

    for document in documents:
        nested_children = document.get("children")
        parent = _document_text(
            document,
            "parent",
            "root_category",
            "category_group",
            "family",
        ) or "Other"

        if isinstance(nested_children, list):
            candidates = [child for child in nested_children if isinstance(child, dict)]
        else:
            candidates = [document]

        for candidate in candidates:
            value = _document_text(candidate, "value", "category", "type", "slug", "_id")
            if not value:
                continue
            label = (
                _document_text(candidate, "label", "category_cn", "name", "display_name")
                or value
            )
            grouped.setdefault(parent, {})[value] = CategoryOption(label=label, value=value)

    return [
        CategoryGroup(parent=parent, children=list(options.values()))
        for parent, options in sorted(grouped.items(), key=lambda item: item[0].casefold())
    ]


async def get_categories(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> StudioCategories:
    project = await database.projects.find_one(
        {"_id": project_id, "owner_id": principal.user_id}, {"_id": 1}
    )
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)

    devices = await database.devices.find({}).to_list(None)
    sensors = await database.sensors.find({}).to_list(None)
    return StudioCategories(
        devices=_normalize_categories(devices),
        sensors=_normalize_categories(sensors),
    )


def validate_graph(request: StudioGraphUpdate) -> None:
    node_ids = [node.id for node in request.nodes]
    edge_ids = [edge.id for edge in request.edges]
    if len(node_ids) != len(set(node_ids)):
        raise AppError("duplicate_node_id", "Graph node IDs must be unique", status_code=422)
    if len(edge_ids) != len(set(edge_ids)):
        raise AppError("duplicate_edge_id", "Graph edge IDs must be unique", status_code=422)
    known_nodes = set(node_ids)
    nodes_by_id = {node.id: node for node in request.nodes}
    invalid_edges = [
        edge.id
        for edge in request.edges
        if edge.source not in known_nodes or edge.target not in known_nodes
    ]
    if invalid_edges:
        raise AppError(
            "invalid_edge_endpoint",
            "Every graph edge must reference existing nodes",
            status_code=422,
            details={"edge_ids": invalid_edges},
        )
    invalid_parents = [
        node.id
        for node in request.nodes
        if node.parent_id is not None and node.parent_id not in known_nodes
    ]
    if invalid_parents:
        raise AppError(
            "invalid_parent_node",
            "Every parent_id must reference an existing node",
            status_code=422,
            details={"node_ids": invalid_parents},
        )
    invalid_parent_types = [
        node.id
        for node in request.nodes
        if node.parent_id is not None
        and node.parent_id in nodes_by_id
        and nodes_by_id[node.parent_id].type != "group"
    ]
    if invalid_parent_types:
        raise AppError(
            "invalid_parent_type",
            "A graph parent must be a group node",
            status_code=422,
            details={"node_ids": invalid_parent_types},
        )
    invalid_handle_edges = [
        edge.id
        for edge in request.edges
        if (
            edge.data.get("sourceHandle") is not None
            and not str(edge.data["sourceHandle"]).startswith("source-")
        )
        or (
            edge.data.get("targetHandle") is not None
            and not str(edge.data["targetHandle"]).startswith("target-")
        )
    ]
    if invalid_handle_edges:
        raise AppError(
            "invalid_edge_handle",
            "Connections must run from a source handle to a target handle",
            status_code=422,
            details={"edge_ids": invalid_handle_edges},
        )
    for node in request.nodes:
        visited = {node.id}
        parent_id = node.parent_id
        while parent_id is not None:
            if parent_id in visited:
                raise AppError(
                    "cyclic_node_parent",
                    "Graph groups cannot contain themselves directly or indirectly",
                    status_code=422,
                    details={"node_id": node.id},
                )
            visited.add(parent_id)
            parent_id = nodes_by_id[parent_id].parent_id
    self_edges = [edge.id for edge in request.edges if edge.source == edge.target]
    if self_edges:
        raise AppError(
            "self_edge",
            "Graph edges cannot connect a node to itself",
            status_code=422,
            details={"edge_ids": self_edges},
        )


async def get_graph(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> StudioGraph:
    project = await database.projects.find_one({"_id": project_id, "owner_id": principal.user_id})
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return StudioGraph(
        project_id=project_id,
        revision=int(project.get("graph_revision", 0)),
        nodes=project.get("nodes", []),
        edges=project.get("edges", []),
        updated_at=project["updated_at"],
    )


async def save_graph(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: StudioGraphUpdate,
) -> StudioGraph:
    validate_graph(request)
    now = datetime.now(UTC)
    next_revision = request.revision + 1
    nodes = [node.model_dump(mode="json") for node in request.nodes]
    edges = [edge.model_dump(mode="json") for edge in request.edges]
    project = await database.projects.find_one_and_update(
        {
            "_id": project_id,
            "owner_id": principal.user_id,
            "graph_revision": request.revision,
        },
        {
            "$set": {
                "nodes": nodes,
                "edges": edges,
                "graph_revision": next_revision,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if project is None:
        owned = await database.projects.find_one(
            {"_id": project_id, "owner_id": principal.user_id}, {"graph_revision": 1}
        )
        if owned is None:
            raise AppError("project_not_found", "Project was not found", status_code=404)
        raise AppError(
            "graph_revision_conflict",
            "The Studio graph changed after it was loaded",
            status_code=409,
            details={"current_revision": int(owned.get("graph_revision", 0))},
        )
    await database.studio_graph_versions.update_one(
        {"project_id": project_id, "revision": next_revision},
        {
            "$setOnInsert": {
                "_id": f"{project_id}:{next_revision}",
                "project_id": project_id,
                "owner_id": principal.user_id,
                "revision": next_revision,
                "nodes": nodes,
                "edges": edges,
                "created_at": now,
            }
        },
        upsert=True,
    )
    return StudioGraph(
        project_id=project_id,
        revision=next_revision,
        nodes=nodes,
        edges=edges,
        updated_at=now,
    )
