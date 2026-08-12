from __future__ import annotations

import json
from typing import Any

import pytest

import app.modules.agents.tools.project as project_tools_module
from app.core.errors import AppError
from app.modules.agents.tools.base import ToolContext
from app.modules.agents.tools.project import project_tools
from app.modules.agents.tools.studio import studio_tools
from app.modules.projects.schemas import DataQueryResult, PropertyCatalog


class FakeProjects:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        if query == {"_id": "prj_test", "owner_id": "usr_test"}:
            return self.document
        return None


class FakeDatabase:
    def __init__(self) -> None:
        self.projects = FakeProjects(
            {
                "_id": "prj_test",
                "owner_id": "usr_test",
                "name": "Cooling Plant",
                "nodes": [
                    {
                        "id": "chiller-1",
                        "type": "equipment",
                        "data": {
                            "label": "CH-1",
                            "name": "remote-chiller-001",
                            "category": "Centrifugal_Chiller",
                            "sensors": [
                                {
                                    "id": "sat-1",
                                    "name": "Supply temperature",
                                    "category": "Temperature_Sensor",
                                }
                            ],
                        },
                    }
                ],
                "edges": [],
            }
        )


CONTEXT = ToolContext("run_test", "ses_test", "prj_test", "usr_test")


@pytest.mark.asyncio
async def test_project_tools_expose_rdf_instead_of_studio_graph() -> None:
    tools = {tool.name: tool for tool in project_tools(FakeDatabase())}  # type: ignore[arg-type]

    assert "get_project_graph" not in tools
    assert "get_project_rdf" in tools
    assert "query_project_rdf" in tools
    assert "get_project_graph" in {
        tool.name
        for tool in studio_tools(FakeDatabase())  # type: ignore[arg-type]
    }

    result = await tools["get_project_rdf"].execute({}, CONTEXT)
    assert result.details == {"media_type": "text/turtle"}
    assert "@prefix brick:" in result.content
    assert "enerai:CH-1 a brick:Centrifugal_Chiller" in result.content


@pytest.mark.asyncio
async def test_rdf_query_tool_runs_select_and_limits_results() -> None:
    tools = {tool.name: tool for tool in project_tools(FakeDatabase())}  # type: ignore[arg-type]
    result = await tools["query_project_rdf"].execute(
        {
            "query": (
                "PREFIX brick: <https://brickschema.org/schema/Brick#> "
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                "SELECT ?entity ?label WHERE { "
                "?entity a brick:Centrifugal_Chiller ; rdfs:label ?label . }"
            ),
            "limit": 10,
        },
        CONTEXT,
    )
    payload = json.loads(result.content)

    assert payload["type"] == "SELECT"
    assert payload["returned"] == 1
    assert payload["truncated"] is False
    assert payload["bindings"][0]["label"] == {"type": "literal", "value": "CH-1"}

    ask_result = await tools["query_project_rdf"].execute(
        {
            "query": (
                "PREFIX brick: <https://brickschema.org/schema/Brick#> "
                "ASK { ?entity a brick:Centrifugal_Chiller . }"
            )
        },
        CONTEXT,
    )
    assert json.loads(ask_result.content) == {"type": "ASK", "boolean": True}


@pytest.mark.asyncio
async def test_rdf_query_tool_rejects_non_read_query_forms() -> None:
    tools = {tool.name: tool for tool in project_tools(FakeDatabase())}  # type: ignore[arg-type]

    with pytest.raises(AppError) as captured:
        await tools["query_project_rdf"].execute(
            {"query": "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"},
            CONTEXT,
        )

    assert captured.value.code == "unsupported_rdf_query"


@pytest.mark.asyncio
async def test_device_data_tools_follow_remote_data_api_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    property_calls: list[list[str] | None] = []
    query_calls = []

    async def fake_properties(
        _: Any, __: Any, ___: str, device_ids: list[str] | None = None
    ) -> PropertyCatalog:
        property_calls.append(device_ids)
        return PropertyCatalog(
            items=[{"device_id": "remote-chiller-001", "name": "temperature", "unit": "°C"}],
            total=1,
        )

    async def fake_query_data(_: Any, __: Any, ___: str, request: Any) -> DataQueryResult:
        query_calls.append(request)
        return DataQueryResult(data={"device_id": request.device_id, "points": []})

    monkeypatch.setattr(project_tools_module, "properties", fake_properties)
    monkeypatch.setattr(project_tools_module, "query_data", fake_query_data)
    tools = {tool.name: tool for tool in project_tools(FakeDatabase())}  # type: ignore[arg-type]

    assert "get_project_properties" not in tools
    assert "query_project_data" not in tools
    assert tools["get_project_device_properties"].input_schema["required"] == ["label"]
    assert tools["query_project_device_data"].input_schema["required"] == [
        "label",
        "start_time",
        "end_time",
    ]

    catalog_result = await tools["get_project_device_properties"].execute(
        {"label": "CH-1"}, CONTEXT
    )
    assert json.loads(catalog_result.content) == {
        "label": "CH-1",
        "device_id": "remote-chiller-001",
        "properties": [{"device_id": "remote-chiller-001", "name": "temperature", "unit": "°C"}],
        "total": 1,
    }
    assert property_calls == [["remote-chiller-001"]]

    query_result = await tools["query_project_device_data"].execute(
        {
            "label": "CH-1",
            "properties": ["temperature"],
            "start_time": "2026-08-12T00:00:00+00:00",
            "end_time": "2026-08-12T01:00:00+00:00",
        },
        CONTEXT,
    )
    assert json.loads(query_result.content) == {
        "data": {"device_id": "remote-chiller-001", "points": []}
    }
    assert len(query_calls) == 1
    assert query_calls[0].device_id == "remote-chiller-001"
    assert query_calls[0].properties == ["temperature"]
    assert query_calls[0].start_time.isoformat() == "2026-08-12T00:00:00+00:00"
    assert query_calls[0].end_time.isoformat() == "2026-08-12T01:00:00+00:00"


@pytest.mark.asyncio
async def test_device_data_tool_rejects_ambiguous_device_labels() -> None:
    database = FakeDatabase()
    duplicate = {
        **database.projects.document["nodes"][0],
        "id": "chiller-2",
        "data": {**database.projects.document["nodes"][0]["data"]},
    }
    database.projects.document["nodes"].append(duplicate)
    tools = {tool.name: tool for tool in project_tools(database)}  # type: ignore[arg-type]

    with pytest.raises(AppError) as captured:
        await tools["get_project_device_properties"].execute({"label": "CH-1"}, CONTEXT)

    assert captured.value.code == "ambiguous_project_node_label"
