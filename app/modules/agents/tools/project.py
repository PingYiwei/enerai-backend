from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, cast

from pymongo.asynchronous.database import AsyncDatabase
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.query import ResultRow

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.runtime.types import JsonObject, ToolResult
from app.modules.agents.tools.base import Tool, ToolContext
from app.modules.projects.data import owned_project, project_rdf, properties, query_data
from app.modules.projects.schemas import DataQuery

Document = dict[str, Any]
_QUERY_FORM = re.compile(
    r"\A\s*(?:(?:PREFIX\s+(?:[A-Za-z][\w.-]*)?:\s*<[^>]+>|BASE\s*<[^>]+>)\s*)*"
    r"([A-Za-z]+)\b",
    re.IGNORECASE | re.DOTALL,
)


def _rdf_term(value: Any) -> JsonObject:
    if isinstance(value, URIRef):
        return {"type": "uri", "value": str(value)}
    if isinstance(value, Literal):
        result: JsonObject = {"type": "literal", "value": str(value)}
        if value.datatype is not None:
            result["datatype"] = str(value.datatype)
        if value.language is not None:
            result["language"] = value.language
        return result
    if isinstance(value, BNode):
        return {"type": "bnode", "value": str(value)}
    return {"type": "unknown", "value": str(value)}


def _validate_query_form(query: str) -> None:
    without_comments = re.sub(r"(?m)^\s*#.*$", "", query)
    match = _QUERY_FORM.match(without_comments)
    if match is None or match.group(1).upper() not in {"SELECT", "ASK"}:
        raise AppError(
            "unsupported_rdf_query",
            "Only read-only SPARQL SELECT and ASK queries are supported",
            status_code=422,
        )


def project_tools(database: AsyncDatabase[Document]) -> tuple[Tool, ...]:
    async def get_rdf(_: JsonObject, context: ToolContext) -> ToolResult:
        project = await owned_project(
            database,
            Principal(user_id=context.user_id, username=""),
            context.project_id,
        )
        return ToolResult(
            tool_call_id="",
            content=project_rdf(project),
            details={"media_type": "text/turtle"},
        )

    async def query_rdf(arguments: JsonObject, context: ToolContext) -> ToolResult:
        sparql = str(arguments["query"]).strip()
        _validate_query_form(sparql)
        limit = int(arguments.get("limit", 200))
        project = await owned_project(
            database,
            Principal(user_id=context.user_id, username=""),
            context.project_id,
        )
        graph = Graph()
        graph.parse(data=project_rdf(project), format="turtle")
        try:
            result = graph.query(sparql, initNs=dict(graph.namespaces()))
        except Exception as error:
            raise AppError(
                "invalid_rdf_query",
                f"SPARQL query is invalid: {error}",
                status_code=422,
            ) from error

        if result.type == "ASK":
            payload: JsonObject = {"type": "ASK", "boolean": bool(result.askAnswer)}
        else:
            variables = [str(variable) for variable in result.vars or []]
            bindings: list[JsonObject] = []
            truncated = False
            for index, row in enumerate(result):
                if index >= limit:
                    truncated = True
                    break
                bindings.append(
                    {
                        str(variable): _rdf_term(value)
                        for variable, value in cast(ResultRow, row).asdict().items()
                        if value is not None
                    }
                )
            payload = {
                "type": "SELECT",
                "variables": variables,
                "bindings": bindings,
                "returned": len(bindings),
                "truncated": truncated,
            }
        return ToolResult(
            tool_call_id="",
            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
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

    async def query_timeseries(arguments: JsonObject, context: ToolContext) -> ToolResult:
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
            name="get_project_rdf",
            description=(
                "Read the current project's semantic equipment model as RDF in Turtle format."
            ),
            input_schema={"type": "object", "additionalProperties": False},
            execute=get_rdf,
            effect="read",
            result_visibility="both",
        ),
        Tool(
            name="query_project_rdf",
            description=(
                "Run a read-only SPARQL SELECT or ASK query against the current project's "
                "RDF model."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 20_000},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1_000},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            execute=query_rdf,
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
            execute=query_timeseries,
            effect="external",
            result_visibility="model",
        ),
    )
