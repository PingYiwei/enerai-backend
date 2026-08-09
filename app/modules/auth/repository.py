from __future__ import annotations

from typing import Any, Protocol

from pymongo.asynchronous.database import AsyncDatabase

Document = dict[str, Any]


class UserRepository(Protocol):
    async def insert(self, document: Document) -> None: ...

    async def find_by_username_key(self, username_key: str) -> Document | None: ...


class MongoUserRepository:
    def __init__(self, database: AsyncDatabase[Document]) -> None:
        self._collection = database.users

    async def insert(self, document: Document) -> None:
        await self._collection.insert_one(document)

    async def find_by_username_key(self, username_key: str) -> Document | None:
        return await self._collection.find_one({"username_key": username_key})
