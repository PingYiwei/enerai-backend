from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

from pymongo.asynchronous.database import AsyncDatabase

from app.core.security import Principal
from app.modules.projects.repository import ProjectRepository
from app.modules.projects.service import project_token_usage


class FakeRepository:
    async def get_for_owner(self, project_id: str, owner_id: str) -> dict[str, Any]:
        return {"_id": project_id, "owner_id": owner_id}


class FakeCursor:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document

    async def to_list(self, _: int) -> list[dict[str, Any]]:
        return [self.document]


class FakeOperations:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document
        self.pipeline: list[dict[str, Any]] | None = None

    async def aggregate(self, pipeline: list[dict[str, Any]]) -> FakeCursor:
        self.pipeline = pipeline
        return FakeCursor(self.document)


class FakeDatabase:
    def __init__(self, document: dict[str, Any]) -> None:
        self.agent_operations = FakeOperations(document)


async def test_project_token_usage_groups_modules_and_fills_missing_days() -> None:
    local_today = (datetime.now(UTC) + timedelta(hours=8)).date()
    database = FakeDatabase(
        {
            "by_module": [
                {"_id": "insight", "input_tokens": 10, "output_tokens": 4},
                {"_id": "studio", "input_tokens": 20, "output_tokens": 5},
            ],
            "daily": [
                {
                    "_id": local_today.isoformat(),
                    "input_tokens": 30,
                    "output_tokens": 9,
                }
            ],
        }
    )

    result = await project_token_usage(
        cast(ProjectRepository, FakeRepository()),
        cast(AsyncDatabase[dict[str, Any]], database),
        Principal(user_id="usr_1", username="test"),
        "prj_1",
        days=3,
        timezone_offset_minutes=480,
    )

    assert result.today == local_today
    assert result.today_total_tokens == 39
    assert [(item.module, item.total_tokens) for item in result.by_module] == [
        ("insight", 14),
        ("studio", 25),
        ("inspection", 0),
    ]
    assert [item.total_tokens for item in result.daily] == [0, 0, 39]
    assert database.agent_operations.pipeline is not None
    assert (
        database.agent_operations.pipeline[1]["$facet"]["daily"][0]["$group"]["_id"][
            "$dateToString"
        ]["timezone"]
        == "+08:00"
    )
