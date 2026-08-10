from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

import app.modules.agents.studio_tools as studio_tools_module
from app.modules.agents.studio_tools import studio_tools
from app.modules.agents.tools import ToolContext
from app.modules.studio.schemas import StudioGraph, StudioGraphUpdate
from app.modules.studio.service import validate_graph

CONTEXT = ToolContext("run_test", "ses_test", "prj_test", "usr_test")


@pytest.mark.asyncio
async def test_studio_tools_expose_atomic_node_sensor_and_edge_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "graph": StudioGraph(
            project_id="prj_test",
            revision=3,
            nodes=[
                {
                    "id": "group-1",
                    "type": "group",
                    "position": {"x": 0, "y": 0},
                    "data": {"child": ["pump-1"]},
                    "parent_id": None,
                },
                {
                    "id": "pump-1",
                    "type": "pump",
                    "position": {"x": 40, "y": 40},
                    "data": {"label": "P-1", "sensors": []},
                    "parent_id": "group-1",
                },
            ],
            edges=[],
            updated_at=datetime.now(UTC),
        )
    }

    async def fake_get_graph(*_: Any) -> StudioGraph:
        return state["graph"].model_copy(deep=True)

    async def fake_save_graph(
        _database: Any, _principal: Any, project_id: str, request: StudioGraphUpdate
    ) -> StudioGraph:
        validate_graph(request)
        saved = StudioGraph(
            project_id=project_id,
            revision=request.revision + 1,
            nodes=request.nodes,
            edges=request.edges,
            updated_at=datetime.now(UTC),
        )
        state["graph"] = saved
        return saved.model_copy(deep=True)

    monkeypatch.setattr(studio_tools_module, "get_graph", fake_get_graph)
    monkeypatch.setattr(studio_tools_module, "save_graph", fake_save_graph)
    tools = {tool.name: tool for tool in studio_tools(object())}  # type: ignore[arg-type]

    assert "replace_studio_graph" not in tools
    assert set(tools) == {
        "get_project_graph",
        "create_studio_node",
        "update_studio_node",
        "delete_studio_node",
        "create_studio_sensor",
        "update_studio_sensor",
        "delete_studio_sensor",
        "create_studio_edge",
        "update_studio_edge",
        "delete_studio_edge",
    }
    assert all(
        tool.execution_mode == "sequential"
        for name, tool in tools.items()
        if name != "get_project_graph"
    )

    created_sensor = await tools["create_studio_sensor"].execute(
        {
            "node_id": "pump-1",
            "sensor": {
                "id": "sensor-1",
                "name": "Discharge pressure",
                "category": "Pressure_Sensor",
            },
        },
        CONTEXT,
    )
    assert json.loads(created_sensor.content)["revision"] == 4

    updated_sensor = await tools["update_studio_sensor"].execute(
        {
            "node_id": "pump-1",
            "sensor_id": "sensor-1",
            "description": "Pump discharge pressure",
        },
        CONTEXT,
    )
    assert json.loads(updated_sensor.content)["sensor"]["description"] == (
        "Pump discharge pressure"
    )

    await tools["update_studio_node"].execute(
        {"node_id": "pump-1", "parent_id": None}, CONTEXT
    )
    assert state["graph"].nodes[0].data["child"] == []

    await tools["create_studio_node"].execute(
        {
            "id": "chiller-1",
            "type": "chiller",
            "position": {"x": 240, "y": 40},
            "data": {"label": "CH-1", "sensors": []},
        },
        CONTEXT,
    )
    await tools["create_studio_edge"].execute(
        {
            "id": "edge-1",
            "source": "pump-1",
            "target": "chiller-1",
            "data": {"sourceHandle": "source-0", "targetHandle": "target-0"},
        },
        CONTEXT,
    )
    deleted = await tools["delete_studio_node"].execute(
        {"node_id": "group-1"}, CONTEXT
    )

    assert json.loads(deleted.content)["revision"] == 9
    assert state["graph"].nodes[0].id == "pump-1"
    assert state["graph"].nodes[0].parent_id is None
    assert [edge.id for edge in state["graph"].edges] == ["edge-1"]


@pytest.mark.asyncio
async def test_deleting_node_also_deletes_connected_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = StudioGraph(
        project_id="prj_test",
        revision=0,
        nodes=[
            {
                "id": "source",
                "position": {"x": 0, "y": 0},
                "data": {},
                "parent_id": None,
            },
            {
                "id": "target",
                "position": {"x": 100, "y": 0},
                "data": {},
                "parent_id": None,
            },
        ],
        edges=[
            {
                "id": "edge-1",
                "source": "source",
                "target": "target",
                "data": {},
            }
        ],
        updated_at=datetime.now(UTC),
    )

    async def fake_get_graph(*_: Any) -> StudioGraph:
        return graph.model_copy(deep=True)

    async def fake_save_graph(
        _database: Any, _principal: Any, project_id: str, request: StudioGraphUpdate
    ) -> StudioGraph:
        validate_graph(request)
        return StudioGraph(
            project_id=project_id,
            revision=request.revision + 1,
            nodes=request.nodes,
            edges=request.edges,
            updated_at=datetime.now(UTC),
        )

    monkeypatch.setattr(studio_tools_module, "get_graph", fake_get_graph)
    monkeypatch.setattr(studio_tools_module, "save_graph", fake_save_graph)
    tools = {tool.name: tool for tool in studio_tools(object())}  # type: ignore[arg-type]

    result = await tools["delete_studio_node"].execute({"node_id": "source"}, CONTEXT)
    assert json.loads(result.content)["removed_edge_ids"] == ["edge-1"]
