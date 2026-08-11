from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.errors import AppError
from app.modules.agents.tools.base import ToolContext
from app.modules.agents.tools.project import project_tools
from app.modules.agents.tools.studio import studio_tools


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
                            "name": "CH-1",
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
        tool.name for tool in studio_tools(FakeDatabase())  # type: ignore[arg-type]
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
