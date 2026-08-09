from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, IndexModel
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings

Document = dict[str, Any]


async def create_indexes(database: AsyncDatabase[Document]) -> None:
    await database.users.create_indexes(
        [
            IndexModel([("username_key", ASCENDING)], unique=True),
            IndexModel([("email", ASCENDING)], unique=True),
        ]
    )
    await database.user_api_keys.create_indexes(
        [
            IndexModel([("key_hash", ASCENDING)], unique=True),
            IndexModel([("owner_id", ASCENDING), ("created_at", DESCENDING)]),
        ]
    )
    await database.projects.create_indexes(
        [
            IndexModel([("owner_id", ASCENDING), ("updated_at", DESCENDING)]),
            IndexModel([("owner_id", ASCENDING), ("name_key", ASCENDING)], unique=True),
        ]
    )
    await database.agent_sessions.create_indexes(
        [
            IndexModel(
                [
                    ("owner_id", ASCENDING),
                    ("project_id", ASCENDING),
                    ("updated_at", DESCENDING),
                ]
            ),
        ]
    )
    await database.studio_graph_versions.create_indexes(
        [IndexModel([("project_id", ASCENDING), ("revision", ASCENDING)], unique=True)]
    )
    await database.agent_lanes.create_indexes(
        [IndexModel([("session_id", ASCENDING), ("name", ASCENDING)], unique=True)]
    )
    await database.agent_entries.create_indexes(
        [
            IndexModel([("session_id", ASCENDING), ("seq", ASCENDING)], unique=True),
            IndexModel([("session_id", ASCENDING), ("parent_id", ASCENDING)]),
        ]
    )
    await database.agent_operations.create_indexes(
        [
            IndexModel(
                [
                    ("owner_id", ASCENDING),
                    ("session_id", ASCENDING),
                    ("created_at", DESCENDING),
                ]
            )
        ]
    )
    await database.agent_events.create_indexes(
        [IndexModel([("run_id", ASCENDING), ("seq", ASCENDING)], unique=True)]
    )
    await database.agent_records.create_indexes(
        [IndexModel([("operation_id", ASCENDING), ("seq", ASCENDING)], unique=True)]
    )
    await database.chat_attachments.create_indexes(
        [
            IndexModel(
                [("owner_id", ASCENDING), ("project_id", ASCENDING), ("created_at", DESCENDING)]
            ),
            IndexModel([("status", ASCENDING), ("created_at", ASCENDING)]),
        ]
    )
    await database.artifacts.create_indexes(
        [
            IndexModel(
                [("owner_id", ASCENDING), ("session_id", ASCENDING), ("created_at", DESCENDING)]
            ),
            IndexModel([("project_id", ASCENDING), ("created_at", DESCENDING)]),
        ]
    )
    await database.inspection_policies.create_indexes(
        [IndexModel([("project_id", ASCENDING)], unique=True)]
    )
    await database.inspection_runs.create_indexes(
        [
            IndexModel(
                [
                    ("owner_id", ASCENDING),
                    ("project_id", ASCENDING),
                    ("started_at", DESCENDING),
                ]
            ),
            IndexModel(
                [("project_id", ASCENDING), ("schedule_slot", ASCENDING)],
                unique=True,
                sparse=True,
            ),
        ]
    )
    await database.datasets.create_indexes(
        [
            IndexModel(
                [("owner_id", ASCENDING), ("project_id", ASCENDING), ("created_at", DESCENDING)]
            )
        ]
    )
    await database.models.create_indexes(
        [
            IndexModel(
                [("owner_id", ASCENDING), ("project_id", ASCENDING), ("created_at", DESCENDING)]
            )
        ]
    )


@asynccontextmanager
async def database_lifespan(settings: Settings) -> AsyncIterator[AsyncDatabase[Document]]:
    client: AsyncMongoClient[Document] = AsyncMongoClient(settings.mongodb_uri)
    database = client.get_database(settings.mongodb_database)
    await client.admin.command("ping")
    await create_indexes(database)
    try:
        yield database
    finally:
        await client.close()
