from __future__ import annotations

from typing import Any, cast

from pymongo.asynchronous.database import AsyncDatabase

from app.core.security import Principal
from app.modules.agents.prompts import render_agent_system_prompt
from app.modules.inspections.planning import plan_inspection
from app.modules.inspections.schemas import InspectionRunCreate


class FakeCollection:
    def __init__(self, document: dict[str, Any] | None = None) -> None:
        self.document = document

    async def find_one(
        self, query: dict[str, Any], projection: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        del query, projection
        return self.document


class FakeDatabase:
    def __init__(self) -> None:
        self.projects = FakeCollection(
            {
                "_id": "prj_test",
                "owner_id": "usr_test",
                "graph_revision": 0,
                "nodes": [
                    {
                        "id": "device-1",
                        "type": "chiller",
                        "data": {"label": "L-02", "inspection": {"grade": "B"}},
                    }
                ],
                "edges": [],
            }
        )
        self.studio_graph_versions = FakeCollection()


async def test_temporary_assignment_starts_with_agent_planning_graph() -> None:
    database = cast(AsyncDatabase[dict[str, Any]], FakeDatabase())

    _, manifest, graph, template_name = await plan_inspection(
        database,
        Principal(user_id="usr_test", username="tester"),
        "prj_test",
        InspectionRunCreate(trigger="assignment", instruction="Compare system relationships"),
    )

    assert template_name == "Temporary assignment"
    assert manifest.template_id == "temporary_assignment"
    assert manifest.devices == []
    assert [node.id for node in graph.nodes] == [
        "stage:planning",
        "stage:execution",
        "stage:report",
    ]


def test_temporary_assignment_uses_independent_agent_prompt() -> None:
    prompt = render_agent_system_prompt("assignment")

    assert "Temporary Assignment Agent" in prompt
    assert "smallest useful scope" in prompt
