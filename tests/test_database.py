from __future__ import annotations

from typing import Any, cast

import pytest
from pymongo import IndexModel
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import OperationFailure

from app.core.database import OBSOLETE_INSPECTION_SCHEDULE_INDEX, create_indexes


class FakeIndexCollection:
    def __init__(self, *, drop_error_code: int | None = None) -> None:
        self.created_indexes: list[IndexModel] = []
        self.dropped_indexes: list[str] = []
        self.drop_error_code = drop_error_code

    async def create_indexes(self, indexes: list[IndexModel]) -> None:
        self.created_indexes.extend(indexes)

    async def drop_index(self, name: str) -> None:
        self.dropped_indexes.append(name)
        if self.drop_error_code is not None:
            raise OperationFailure("index not found", code=self.drop_error_code)


class FakeIndexDatabase:
    def __init__(self, *, drop_error_code: int | None = None) -> None:
        self.collections: dict[str, FakeIndexCollection] = {}
        self.drop_error_code = drop_error_code

    def __getattr__(self, name: str) -> FakeIndexCollection:
        collection = self.collections.get(name)
        if collection is None:
            collection = FakeIndexCollection(
                drop_error_code=self.drop_error_code if name == "inspection_runs" else None
            )
            self.collections[name] = collection
        return collection


async def test_create_indexes_drops_obsolete_inspection_schedule_index() -> None:
    database = FakeIndexDatabase()

    await create_indexes(cast(AsyncDatabase[dict[str, Any]], database))

    inspection_runs = database.collections["inspection_runs"]
    assert inspection_runs.dropped_indexes == [OBSOLETE_INSPECTION_SCHEDULE_INDEX]
    assert all(
        "schedule_slot" not in dict(index.document["key"])
        for index in inspection_runs.created_indexes
    )


@pytest.mark.parametrize("error_code", [26, 27])
async def test_create_indexes_accepts_missing_collection_or_obsolete_index(error_code: int) -> None:
    database = FakeIndexDatabase(drop_error_code=error_code)

    await create_indexes(cast(AsyncDatabase[dict[str, Any]], database))

    assert database.collections["inspection_runs"].created_indexes
