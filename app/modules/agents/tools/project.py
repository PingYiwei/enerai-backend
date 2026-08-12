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


async def _context_project(database: AsyncDatabase[Document], context: ToolContext) -> Document:
    return await owned_project(
        database,
        Principal(user_id=context.user_id, username=""),
        context.project_id,
    )


def _node_device(project: Document, label: str) -> str:
    matches: list[str] = []
    for node in project.get("nodes", []):
        data = node.get("data")
        if not isinstance(data, dict):
            continue
        node_label = str(data.get("label") or data.get("name") or node.get("id") or "").strip()
        if node_label != label:
            continue
        device_id = str(data.get("name") or data.get("label") or "").strip()
        if device_id:
            matches.append(device_id)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AppError(
            "ambiguous_project_node_label",
            f'Multiple project nodes use the RDF label "{label}"',
            status_code=422,
        )
    raise AppError(
        "project_node_not_found",
        "No project node with this RDF label has a data-source device name",
        status_code=404,
    )


def project_tools(database: AsyncDatabase[Document]) -> tuple[Tool, ...]:
    async def get_rdf(_: JsonObject, context: ToolContext) -> ToolResult:
        project = await _context_project(database, context)
        return ToolResult(
            tool_call_id="",
            content=project_rdf(project),
            details={"media_type": "text/turtle"},
        )

    async def query_rdf(arguments: JsonObject, context: ToolContext) -> ToolResult:
        sparql = str(arguments["query"]).strip()
        _validate_query_form(sparql)
        limit = int(arguments.get("limit", 200))
        project = await _context_project(database, context)
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

    async def get_device_properties(arguments: JsonObject, context: ToolContext) -> ToolResult:
        label = str(arguments["label"]).strip()
        project = await _context_project(database, context)
        device_id = _node_device(project, label)
        catalog = await properties(
            database,
            Principal(user_id=context.user_id, username=""),
            context.project_id,
            device_ids=[device_id],
        )
        return ToolResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "label": label,
                    "device_id": device_id,
                    "properties": catalog.items,
                    "total": catalog.total,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    async def query_device_data(arguments: JsonObject, context: ToolContext) -> ToolResult:
        label = str(arguments["label"]).strip()
        project = await _context_project(database, context)
        device_id = _node_device(project, label)
        property_values = arguments.get("properties")
        request = DataQuery(
            device_id=device_id,
            properties=[str(item) for item in property_values]
            if isinstance(property_values, list)
            else None,
            start_time=datetime.fromisoformat(str(arguments["start_time"])),
            end_time=datetime.fromisoformat(str(arguments["end_time"])),
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
            name="get_project_device_properties",
            description=(
                "List properties available from the operational data source for one Reality Model "
                "node. Pass its exact device label from an explicit user reference or the project "
                "RDF. Use this before querying the node's time-series data."
            ),
            input_schema={
                "type": "object",
                "properties": {"label": {"type": "string", "minLength": 1, "maxLength": 500}},
                "required": ["label"],
                "additionalProperties": False,
            },
            execute=get_device_properties,
            effect="external",
            result_visibility="model",
        ),
        Tool(
            name="query_project_device_data",
            description=(
                "Query time-series data for one Reality Model node and a bounded ISO-8601 "
                "time range. Pass its exact device label and call "
                "get_project_device_properties first to choose properties."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "minLength": 1, "maxLength": 500},
                    "properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 200,
                    },
                    "start_time": {"type": "string", "format": "date-time"},
                    "end_time": {"type": "string", "format": "date-time"},
                },
                "required": ["label", "start_time", "end_time"],
                "additionalProperties": False,
            },
            execute=query_device_data,
            effect="external",
            result_visibility="model",
        ),
    )
