from __future__ import annotations

import json
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.security import Principal
from app.modules.agents.tools import Tool, ToolContext
from app.modules.agents.types import JsonObject, ToolResult
from app.modules.studio.schemas import StudioGraphUpdate
from app.modules.studio.service import save_graph

Document = dict[str, Any]


def studio_tools(database: AsyncDatabase[Document]) -> tuple[Tool, ...]:
    async def replace_graph(arguments: JsonObject, context: ToolContext) -> ToolResult:
        graph = await save_graph(
            database,
            Principal(user_id=context.user_id, username=""),
            context.project_id,
            StudioGraphUpdate.model_validate(arguments),
        )
        return ToolResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "project_id": graph.project_id,
                    "revision": graph.revision,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                },
                separators=(",", ":"),
            ),
        )

    return (
        Tool(
            name="replace_studio_graph",
            description=(
                "Replace the complete Studio graph at its current revision. Preserve existing "
                "node positions unless the user explicitly requests layout changes."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "revision": {"type": "integer", "minimum": 0},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "position": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number"},
                                        "y": {"type": "number"},
                                    },
                                    "required": ["x", "y"],
                                    "additionalProperties": False,
                                },
                                "data": {"type": "object"},
                                "parent_id": {"type": ["string", "null"]},
                            },
                            "required": ["id", "type", "position", "data", "parent_id"],
                            "additionalProperties": False,
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                                "type": {"type": "string"},
                                "data": {"type": "object"},
                            },
                            "required": ["id", "source", "target", "type", "data"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["revision", "nodes", "edges"],
                "additionalProperties": False,
            },
            execute=replace_graph,
            effect="write",
            execution_mode="sequential",
            result_visibility="both",
            idempotent=False,
        ),
    )
