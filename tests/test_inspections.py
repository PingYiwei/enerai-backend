from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from pymongo.asynchronous.database import AsyncDatabase

from app.core.security import Principal
from app.modules.inspections.service import _inspection_run, create_run, inspect_graph, list_runs


class FakeCursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, *_: object) -> FakeCursor:
        return self

    async def to_list(self, _: int | None) -> list[dict[str, Any]]:
        return self.documents


class FakeCollection:
    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents or []

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        return next(
            (
                document
                for document in self.documents
                if all(document.get(key) == value for key, value in query.items())
            ),
            None,
        )

    async def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(document)

    def find(self, query: dict[str, Any]) -> FakeCursor:
        return FakeCursor(
            [
                document
                for document in self.documents
                if all(document.get(key) == value for key, value in query.items())
            ]
        )


class FakeDatabase:
    def __init__(self) -> None:
        self.projects = FakeCollection(
            [
                {
                    "_id": "prj_test",
                    "owner_id": "usr_test",
                    "graph_revision": 4,
                    "nodes": [],
                    "edges": [],
                }
            ]
        )
        self.inspection_policies = FakeCollection()
        self.inspection_runs = FakeCollection()


def test_empty_graph_returns_actionable_finding() -> None:
    findings = inspect_graph({"nodes": [], "edges": []})
    assert [finding.code for finding in findings] == ["graph_empty"]


def test_graph_reports_isolated_and_unmapped_sensor_nodes() -> None:
    findings = inspect_graph(
        {
            "nodes": [
                {"id": "pump", "type": "pump", "data": {}},
                {"id": "sensor", "type": "sensor", "data": {}},
            ],
            "edges": [],
        }
    )
    assert {finding.code for finding in findings} == {
        "isolated_equipment",
        "sensor_property_missing",
    }
    assert findings[0].node_ids == ["pump", "sensor"]


def test_graph_uses_embedded_studio_sensors_and_ignores_group_connectivity() -> None:
    findings = inspect_graph(
        {
            "nodes": [
                {"id": "group", "type": "group", "data": {}},
                {
                    "id": "chiller",
                    "type": "equipment",
                    "data": {
                        "sensors": [
                            {
                                "id": "sensor-1",
                                "name": "Supply temperature",
                                "category": "temperature",
                            }
                        ]
                    },
                },
                {
                    "id": "pump",
                    "type": "equipment",
                    "data": {"sensors": [{"id": "sensor-2", "name": "Pressure"}]},
                },
            ],
            "edges": [{"source": "chiller", "target": "pump"}],
        }
    )

    assert len(findings) == 1
    assert findings[0].code == "sensor_property_missing"
    assert findings[0].node_ids == ["pump"]


def test_inspection_run_maps_mongo_id_and_supports_historical_documents() -> None:
    now = datetime.now(UTC)
    run = _inspection_run(
        {
            "_id": "isr_test",
            "project_id": "prj_test",
            "status": "completed",
            "trigger": "manual",
            "graph_revision": 3,
            "findings": [],
            "started_at": now,
            "completed_at": now,
        }
    )

    assert run.id == "isr_test"
    assert run.checks == ["graph_integrity", "sensor_coverage"]


async def test_create_and_list_runs_return_public_ids() -> None:
    database = FakeDatabase()
    typed_database = cast(AsyncDatabase[dict[str, Any]], database)
    principal = Principal(user_id="usr_test", username="tester")

    created = await create_run(typed_database, principal, "prj_test")
    listed = await list_runs(typed_database, principal, "prj_test")

    assert created.id.startswith("isr_")
    assert listed.total == 1
    assert listed.items[0].id == created.id
