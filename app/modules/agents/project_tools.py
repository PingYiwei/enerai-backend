from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.tools import Tool, ToolContext
from app.modules.agents.types import JsonObject, ToolResult
from app.modules.projects.data import properties, query_data
from app.modules.projects.schemas import DataQuery

Document = dict[str, Any]


def project_tools(database: AsyncDatabase[Document]) -> tuple[Tool, ...]:
    async def get_graph(_: JsonObject, context: ToolContext) -> ToolResult:
        project = await database.projects.find_one(
            {"_id": context.project_id, "owner_id": context.user_id},
            {"name": 1, "graph_revision": 1, "nodes": 1, "edges": 1},
        )
        if project is None:
            raise AppError("project_not_found", "Project was not found", status_code=404)
        return ToolResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "name": project["name"],
                    "revision": project.get("graph_revision", 0),
                    "nodes": project.get("nodes", []),
                    "edges": project.get("edges", []),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    async def get_properties(_: JsonObject, context: ToolContext) -> ToolResult:
        catalog = await properties(
            database,
            Principal(user_id=context.user_id, username=""),
            context.project_id,
        )
        return ToolResult(
            tool_call_id="",
            content=catalog.model_dump_json(),
        )

    async def query(arguments: JsonObject, context: ToolContext) -> ToolResult:
        request = DataQuery(
            property_ids=[str(item) for item in arguments["property_ids"]],
            start=datetime.fromisoformat(str(arguments["start"])),
            end=datetime.fromisoformat(str(arguments["end"])),
            limit=int(arguments.get("limit", 10_000)),
        )
        result = await query_data(
            database,
            Principal(user_id=context.user_id, username=""),
            context.project_id,
            request,
        )
        return ToolResult(
            tool_call_id="",
            content=result.model_dump_json(),
        )

    return (
        Tool(
            name="get_project_graph",
            description="Read the current project equipment graph, including node data and edges.",
            input_schema={"type": "object", "additionalProperties": False},
            execute=get_graph,
            effect="read",
            result_visibility="both",
        ),
        Tool(
            name="get_project_properties",
            description="List operational property identifiers exposed by the project data source.",
            input_schema={"type": "object", "additionalProperties": False},
            execute=get_properties,
            effect="external",
            result_visibility="model",
        ),
        Tool(
            name="query_project_data",
            description=(
                "Query project time-series data for property IDs and an ISO-8601 time range."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "property_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 200,
                    },
                    "start": {"type": "string", "format": "date-time"},
                    "end": {"type": "string", "format": "date-time"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100_000},
                },
                "required": ["property_ids", "start", "end"],
                "additionalProperties": False,
            },
            execute=query,
            effect="external",
            result_visibility="model",
        ),
    )
