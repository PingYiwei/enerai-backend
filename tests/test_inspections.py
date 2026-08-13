from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.inspections import planning
from app.modules.inspections.router import event_data_json
from app.modules.inspections.schemas import InspectionRunCreate, InspectionScheduleCreate
from app.modules.inspections.service import (
    _inspection_run,
    create_run,
    delete_run,
    get_run,
    inspect_graph,
    list_runs,
    next_schedule_run,
)
from app.modules.projects.schemas import PropertyCatalog


def test_daily_schedule_uses_the_next_beijing_wall_clock_time() -> None:
    schedule = InspectionScheduleCreate(
        recurrence="day",
        scheduled_time="09:30",
    )

    assert next_schedule_run(schedule, datetime(2026, 8, 13, 2, 0, tzinfo=UTC)) == datetime(
        2026, 8, 14, 1, 30, tzinfo=UTC
    )


def test_weekly_schedule_uses_selected_weekday_and_time() -> None:
    schedule = InspectionScheduleCreate(
        recurrence="week",
        weekday=4,
        scheduled_time="18:00",
    )

    assert next_schedule_run(schedule, datetime(2026, 8, 13, 2, 0, tzinfo=UTC)) == datetime(
        2026, 8, 14, 10, 0, tzinfo=UTC
    )


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

    async def delete_one(self, query: dict[str, Any]) -> None:
        self.documents = [
            document
            for document in self.documents
            if not all(document.get(key) == value for key, value in query.items())
        ]

    async def delete_many(self, query: dict[str, Any]) -> None:
        await self.delete_one(query)

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
                    "nodes": [
                        {
                            "id": "pump-1",
                            "type": "pump",
                            "parent_id": None,
                            "data": {"label": "Pump 1", "inspection": {"grade": "B"}},
                        }
                    ],
                    "edges": [],
                }
            ]
        )
        self.inspection_policies = FakeCollection()
        self.inspection_runs = FakeCollection()
        self.inspection_events: FakeCollection | None = None
        self.inspection_screenings: FakeCollection | None = None
        self.inspection_node_results: FakeCollection | None = None


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


def test_inspection_event_json_serializes_agent_review_datetimes() -> None:
    reviewed_at = datetime(2026, 8, 12, 9, 30, tzinfo=UTC)

    payload = event_data_json({"result": {"reviewed_at": reviewed_at}})

    assert payload == '{"result":{"reviewed_at":"2026-08-12T09:30:00+00:00"}}'


async def test_create_and_list_runs_return_public_ids() -> None:
    database = FakeDatabase()
    typed_database = cast(AsyncDatabase[dict[str, Any]], database)
    principal = Principal(user_id="usr_test", username="tester")

    created = await create_run(typed_database, principal, "prj_test")
    listed = await list_runs(typed_database, principal, "prj_test")

    assert created.id.startswith("isr_")
    assert listed.total == 1
    assert listed.items[0].id == created.id


async def test_delete_run_removes_owned_run_and_related_documents() -> None:
    database = FakeDatabase()
    typed_database = cast(AsyncDatabase[dict[str, Any]], database)
    principal = Principal(user_id="usr_test", username="tester")
    created = await create_run(typed_database, principal, "prj_test")
    database.inspection_events = FakeCollection()
    database.inspection_screenings = FakeCollection()
    database.inspection_node_results = FakeCollection()
    related = {
        "run_id": created.id,
        "project_id": "prj_test",
        "owner_id": "usr_test",
    }
    database.inspection_events.documents.append({"_id": "event", **related})
    database.inspection_screenings.documents.append({"_id": "screening", **related})
    database.inspection_node_results.documents.append({"_id": "result", **related})

    await delete_run(typed_database, principal, created.id, "prj_test")

    assert database.inspection_runs.documents == []
    assert database.inspection_events.documents == []
    assert database.inspection_screenings.documents == []
    assert database.inspection_node_results.documents == []


async def test_critical_plan_uses_grade_threshold_rdf_edges_and_data_premises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = FakeDatabase()
    project = database.projects.documents[0]
    project["nodes"] = [
        {"id": "plant", "type": "group", "data": {"label": "Plant"}},
        {
            "id": "pump-s",
            "type": "pump",
            "parent_id": "plant",
            "data": {
                "label": "Pump S",
                "inspection": {"grade": "S"},
                "sensors": [{"name": "power"}, {"name": "flow"}],
            },
        },
        {
            "id": "pump-a",
            "type": "pump",
            "parent_id": "plant",
            "data": {"label": "Pump A", "inspection": {"grade": "A"}},
        },
        {"id": "pump-b", "type": "pump", "data": {"label": "Pump B"}},
        {
            "id": "pump-c",
            "type": "pump",
            "data": {"label": "Pump C", "inspection": {"grade": "C"}},
        },
    ]
    project["edges"] = [
        {"id": "s-to-a", "source": "pump-s", "target": "pump-a"},
        {"id": "a-to-b", "source": "pump-a", "target": "pump-b"},
    ]

    async def fake_properties(*_: object, **__: object) -> PropertyCatalog:
        return PropertyCatalog(
            items=[
                {"device_id": "Pump S", "name": "power"},
                {"device_id": "Pump A", "name": "temperature"},
            ],
            total=2,
        )

    monkeypatch.setattr(planning, "properties", fake_properties)
    _, manifest, task_graph, _ = await planning.plan_inspection(
        cast(AsyncDatabase[dict[str, Any]], database),
        Principal(user_id="usr_test", username="tester"),
        "prj_test",
        InspectionRunCreate(template_id="critical_equipment"),
    )

    assert manifest.minimum_grade == "A"
    assert [device.node_id for device in manifest.devices] == ["pump-s", "pump-a"]
    pump_s = manifest.devices[0]
    assert pump_s.selected_properties == ["power"]
    assert pump_s.skipped_properties == ["flow"]
    assert "flow" in pump_s.premises[0]
    assert any(edge.relation == "feeds" for edge in task_graph.edges)
    assert any(node.id == "group:plant" for node in task_graph.nodes)
    assert "Data completeness, freshness, and missingness are reportable abnormal dimensions." in (
        manifest.premises
    )


async def test_run_lookup_is_scoped_to_project() -> None:
    database = FakeDatabase()
    typed_database = cast(AsyncDatabase[dict[str, Any]], database)
    principal = Principal(user_id="usr_test", username="tester")
    created = await create_run(typed_database, principal, "prj_test")

    with pytest.raises(AppError) as error:
        await get_run(typed_database, principal, created.id, "prj_other")

    assert error.value.code == "inspection_run_not_found"
