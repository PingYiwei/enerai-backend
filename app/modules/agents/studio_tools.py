from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.tools import Tool, ToolContext
from app.modules.agents.types import JsonObject, ToolResult
from app.modules.studio.schemas import (
    GraphEdge,
    GraphNode,
    StudioGraphUpdate,
    StudioSensor,
)
from app.modules.studio.service import get_graph, save_graph

Document = dict[str, Any]
Mutation = Callable[[list[GraphNode], list[GraphEdge]], JsonObject]

POSITION_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    "required": ["x", "y"],
    "additionalProperties": False,
}
SENSOR_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 160},
        "name": {"type": "string", "minLength": 1, "maxLength": 240},
        "category": {"type": "string", "minLength": 1, "maxLength": 160},
        "category_cn": {"type": ["string", "null"], "maxLength": 240},
        "description": {"type": "string", "maxLength": 2_000},
    },
    "required": ["id", "name", "category"],
    "additionalProperties": False,
}


def _principal(context: ToolContext) -> Principal:
    return Principal(user_id=context.user_id, username="")


def _node(nodes: list[GraphNode], node_id: str) -> GraphNode:
    match = next((node for node in nodes if node.id == node_id), None)
    if match is None:
        raise AppError("studio_node_not_found", "Studio node was not found", status_code=404)
    return match


def _edge(edges: list[GraphEdge], edge_id: str) -> GraphEdge:
    match = next((edge for edge in edges if edge.id == edge_id), None)
    if match is None:
        raise AppError("studio_edge_not_found", "Studio edge was not found", status_code=404)
    return match


def _node_sensors(node: GraphNode) -> list[StudioSensor]:
    raw_sensors = node.data.get("sensors", [])
    if not isinstance(raw_sensors, list):
        raise AppError(
            "invalid_node_sensors",
            "Node sensors must be an array",
            status_code=422,
            details={"node_id": node.id},
        )
    return [StudioSensor.model_validate(sensor) for sensor in raw_sensors]


def _set_node_sensors(node: GraphNode, sensors: list[StudioSensor]) -> None:
    node.data = {
        **node.data,
        "sensors": [sensor.model_dump(mode="json", exclude_none=True) for sensor in sensors],
    }


def _remove_group_child(nodes: list[GraphNode], parent_id: str | None, child_id: str) -> None:
    if parent_id is None:
        return
    parent = _node(nodes, parent_id)
    children = parent.data.get("child")
    if isinstance(children, list):
        parent.data = {**parent.data, "child": [item for item in children if item != child_id]}


def _add_group_child(nodes: list[GraphNode], parent_id: str | None, child_id: str) -> None:
    if parent_id is None:
        return
    parent = _node(nodes, parent_id)
    children = parent.data.get("child")
    child_ids = list(children) if isinstance(children, list) else []
    if child_id not in child_ids:
        parent.data = {**parent.data, "child": [*child_ids, child_id]}


def studio_tools(database: AsyncDatabase[Document]) -> tuple[Tool, ...]:
    async def read_graph(_: JsonObject, context: ToolContext) -> ToolResult:
        graph = await get_graph(database, _principal(context), context.project_id)
        return ToolResult(tool_call_id="", content=graph.model_dump_json())

    async def mutate_graph(
        context: ToolContext, action: str, mutation: Mutation
    ) -> ToolResult:
        current = await get_graph(database, _principal(context), context.project_id)
        nodes = [node.model_copy(deep=True) for node in current.nodes]
        edges = [edge.model_copy(deep=True) for edge in current.edges]
        details = mutation(nodes, edges)
        saved = await save_graph(
            database,
            _principal(context),
            context.project_id,
            StudioGraphUpdate(revision=current.revision, nodes=nodes, edges=edges),
        )
        return ToolResult(
            tool_call_id="",
            content=json.dumps(
                {"action": action, "revision": saved.revision, **details},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    async def create_node(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(nodes: list[GraphNode], _: list[GraphEdge]) -> JsonObject:
            candidate = GraphNode.model_validate(arguments)
            if any(node.id == candidate.id for node in nodes):
                raise AppError(
                    "duplicate_node_id", "Graph node IDs must be unique", status_code=422
                )
            nodes.append(candidate)
            _add_group_child(nodes, candidate.parent_id, candidate.id)
            return {"node": candidate.model_dump(mode="json")}

        return await mutate_graph(context, "node_created", mutation)

    async def update_node(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(nodes: list[GraphNode], _: list[GraphEdge]) -> JsonObject:
            node = _node(nodes, str(arguments["node_id"]))
            previous_parent_id = node.parent_id
            payload = node.model_dump(mode="python")
            for field in ("type", "position", "parent_id"):
                if field in arguments:
                    payload[field] = arguments[field]
            if "data" in arguments:
                payload["data"] = {**node.data, **dict(arguments["data"])}
            updated = GraphNode.model_validate(payload)
            nodes[nodes.index(node)] = updated
            if updated.parent_id != previous_parent_id:
                _remove_group_child(nodes, previous_parent_id, updated.id)
                _add_group_child(nodes, updated.parent_id, updated.id)
            return {"node": updated.model_dump(mode="json")}

        return await mutate_graph(context, "node_updated", mutation)

    async def delete_node(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(nodes: list[GraphNode], edges: list[GraphEdge]) -> JsonObject:
            node_id = str(arguments["node_id"])
            _node(nodes, node_id)
            nodes[:] = [node for node in nodes if node.id != node_id]
            for node in nodes:
                if node.parent_id == node_id:
                    node.parent_id = None
                children = node.data.get("child")
                if isinstance(children, list) and node_id in children:
                    node.data = {
                        **node.data,
                        "child": [child for child in children if child != node_id],
                    }
            removed_edges = [
                edge.id for edge in edges if node_id in {edge.source, edge.target}
            ]
            edges[:] = [
                edge for edge in edges if node_id not in {edge.source, edge.target}
            ]
            return {"node_id": node_id, "removed_edge_ids": removed_edges}

        return await mutate_graph(context, "node_deleted", mutation)

    async def create_sensor(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(nodes: list[GraphNode], _: list[GraphEdge]) -> JsonObject:
            node = _node(nodes, str(arguments["node_id"]))
            sensor = StudioSensor.model_validate(arguments["sensor"])
            sensors = _node_sensors(node)
            if any(candidate.id == sensor.id for candidate in sensors):
                raise AppError(
                    "duplicate_sensor_id",
                    "Sensor IDs must be unique within a node",
                    status_code=422,
                )
            sensors.append(sensor)
            _set_node_sensors(node, sensors)
            return {"node_id": node.id, "sensor": sensor.model_dump(mode="json")}

        return await mutate_graph(context, "sensor_created", mutation)

    async def update_sensor(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(nodes: list[GraphNode], _: list[GraphEdge]) -> JsonObject:
            node = _node(nodes, str(arguments["node_id"]))
            sensor_id = str(arguments["sensor_id"])
            sensors = _node_sensors(node)
            index = next(
                (index for index, sensor in enumerate(sensors) if sensor.id == sensor_id),
                None,
            )
            if index is None:
                raise AppError(
                    "studio_sensor_not_found", "Studio sensor was not found", status_code=404
                )
            payload = sensors[index].model_dump(mode="python")
            for field in ("name", "category", "category_cn", "description"):
                if field in arguments:
                    payload[field] = arguments[field]
            sensors[index] = StudioSensor.model_validate(payload)
            _set_node_sensors(node, sensors)
            return {"node_id": node.id, "sensor": sensors[index].model_dump(mode="json")}

        return await mutate_graph(context, "sensor_updated", mutation)

    async def delete_sensor(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(nodes: list[GraphNode], _: list[GraphEdge]) -> JsonObject:
            node = _node(nodes, str(arguments["node_id"]))
            sensor_id = str(arguments["sensor_id"])
            sensors = _node_sensors(node)
            if not any(sensor.id == sensor_id for sensor in sensors):
                raise AppError(
                    "studio_sensor_not_found", "Studio sensor was not found", status_code=404
                )
            _set_node_sensors(node, [sensor for sensor in sensors if sensor.id != sensor_id])
            return {"node_id": node.id, "sensor_id": sensor_id}

        return await mutate_graph(context, "sensor_deleted", mutation)

    async def create_edge(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(_: list[GraphNode], edges: list[GraphEdge]) -> JsonObject:
            candidate = GraphEdge.model_validate(arguments)
            if any(edge.id == candidate.id for edge in edges):
                raise AppError(
                    "duplicate_edge_id", "Graph edge IDs must be unique", status_code=422
                )
            edges.append(candidate)
            return {"edge": candidate.model_dump(mode="json")}

        return await mutate_graph(context, "edge_created", mutation)

    async def update_edge(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(_: list[GraphNode], edges: list[GraphEdge]) -> JsonObject:
            edge = _edge(edges, str(arguments["edge_id"]))
            payload = edge.model_dump(mode="python")
            for field in ("source", "target", "type"):
                if field in arguments:
                    payload[field] = arguments[field]
            if "data" in arguments:
                payload["data"] = {**edge.data, **dict(arguments["data"])}
            updated = GraphEdge.model_validate(payload)
            edges[edges.index(edge)] = updated
            return {"edge": updated.model_dump(mode="json")}

        return await mutate_graph(context, "edge_updated", mutation)

    async def delete_edge(arguments: JsonObject, context: ToolContext) -> ToolResult:
        def mutation(_: list[GraphNode], edges: list[GraphEdge]) -> JsonObject:
            edge_id = str(arguments["edge_id"])
            _edge(edges, edge_id)
            edges[:] = [edge for edge in edges if edge.id != edge_id]
            return {"edge_id": edge_id}

        return await mutate_graph(context, "edge_deleted", mutation)

    write_options: JsonObject = {
        "effect": "write",
        "execution_mode": "sequential",
        "result_visibility": "both",
        "idempotent": False,
    }
    node_id_property = {"type": "string", "minLength": 1, "maxLength": 160}
    edge_id_property = {"type": "string", "minLength": 1, "maxLength": 160}
    return (
        Tool(
            name="get_project_graph",
            description=(
                "Read the complete current Studio graph, including revision, nodes, and edges."
            ),
            input_schema={"type": "object", "additionalProperties": False},
            execute=read_graph,
            effect="read",
            result_visibility="both",
        ),
        Tool(
            name="create_studio_node",
            description="Create one equipment or group node in the Studio graph.",
            input_schema={
                "type": "object",
                "properties": {
                    "id": node_id_property,
                    "type": {"type": "string", "minLength": 1, "maxLength": 80},
                    "position": POSITION_SCHEMA,
                    "data": {"type": "object"},
                    "parent_id": {"type": ["string", "null"]},
                },
                "required": ["id", "position"],
                "additionalProperties": False,
            },
            execute=create_node,
            **write_options,
        ),
        Tool(
            name="update_studio_node",
            description="Update selected fields of one Studio node; data fields are merged.",
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": node_id_property,
                    "type": {"type": "string", "minLength": 1, "maxLength": 80},
                    "position": POSITION_SCHEMA,
                    "data": {"type": "object"},
                    "parent_id": {"type": ["string", "null"]},
                },
                "required": ["node_id"],
                "additionalProperties": False,
            },
            execute=update_node,
            **write_options,
        ),
        Tool(
            name="delete_studio_node",
            description="Delete one Studio node and its connected edges; child nodes are detached.",
            input_schema={
                "type": "object",
                "properties": {"node_id": node_id_property},
                "required": ["node_id"],
                "additionalProperties": False,
            },
            execute=delete_node,
            **write_options,
        ),
        Tool(
            name="create_studio_sensor",
            description="Create one sensor inside an existing equipment node.",
            input_schema={
                "type": "object",
                "properties": {"node_id": node_id_property, "sensor": SENSOR_SCHEMA},
                "required": ["node_id", "sensor"],
                "additionalProperties": False,
            },
            execute=create_sensor,
            **write_options,
        ),
        Tool(
            name="update_studio_sensor",
            description="Update selected fields of one sensor inside an existing node.",
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": node_id_property,
                    "sensor_id": node_id_property,
                    "name": {"type": "string", "minLength": 1, "maxLength": 240},
                    "category": {"type": "string", "minLength": 1, "maxLength": 160},
                    "category_cn": {"type": ["string", "null"], "maxLength": 240},
                    "description": {"type": "string", "maxLength": 2_000},
                },
                "required": ["node_id", "sensor_id"],
                "additionalProperties": False,
            },
            execute=update_sensor,
            **write_options,
        ),
        Tool(
            name="delete_studio_sensor",
            description="Delete one sensor from an existing Studio node.",
            input_schema={
                "type": "object",
                "properties": {"node_id": node_id_property, "sensor_id": node_id_property},
                "required": ["node_id", "sensor_id"],
                "additionalProperties": False,
            },
            execute=delete_sensor,
            **write_options,
        ),
        Tool(
            name="create_studio_edge",
            description="Create one directed edge between two existing Studio nodes.",
            input_schema={
                "type": "object",
                "properties": {
                    "id": edge_id_property,
                    "source": node_id_property,
                    "target": node_id_property,
                    "type": {"type": "string", "minLength": 1, "maxLength": 80},
                    "data": {"type": "object"},
                },
                "required": ["id", "source", "target"],
                "additionalProperties": False,
            },
            execute=create_edge,
            **write_options,
        ),
        Tool(
            name="update_studio_edge",
            description="Update selected fields of one Studio edge; data fields are merged.",
            input_schema={
                "type": "object",
                "properties": {
                    "edge_id": edge_id_property,
                    "source": node_id_property,
                    "target": node_id_property,
                    "type": {"type": "string", "minLength": 1, "maxLength": 80},
                    "data": {"type": "object"},
                },
                "required": ["edge_id"],
                "additionalProperties": False,
            },
            execute=update_edge,
            **write_options,
        ),
        Tool(
            name="delete_studio_edge",
            description="Delete one edge from the Studio graph.",
            input_schema={
                "type": "object",
                "properties": {"edge_id": edge_id_property},
                "required": ["edge_id"],
                "additionalProperties": False,
            },
            execute=delete_edge,
            **write_options,
        ),
    )
