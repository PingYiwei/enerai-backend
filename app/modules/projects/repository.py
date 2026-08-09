from __future__ import annotations

from typing import Any, Protocol

from pymongo import DESCENDING, ReturnDocument
from pymongo.asynchronous.database import AsyncDatabase

Document = dict[str, Any]


class ProjectRepository(Protocol):
    async def insert(self, document: Document) -> None: ...

    async def list_for_owner(self, owner_id: str) -> list[Document]: ...

    async def get_for_owner(self, project_id: str, owner_id: str) -> Document | None: ...

    async def update_for_owner(
        self,
        project_id: str,
        owner_id: str,
        changes: Document,
    ) -> Document | None: ...

    async def delete_for_owner(self, project_id: str, owner_id: str) -> bool: ...


class MongoProjectRepository:
    def __init__(self, database: AsyncDatabase[Document]) -> None:
        self._collection = database.projects

    async def insert(self, document: Document) -> None:
        await self._collection.insert_one(document)

    async def list_for_owner(self, owner_id: str) -> list[Document]:
        return await (
            self._collection.find({"owner_id": owner_id})
            .sort("updated_at", DESCENDING)
            .to_list(None)
        )

    async def get_for_owner(self, project_id: str, owner_id: str) -> Document | None:
        return await self._collection.find_one({"_id": project_id, "owner_id": owner_id})

    async def update_for_owner(
        self,
        project_id: str,
        owner_id: str,
        changes: Document,
    ) -> Document | None:
        return await self._collection.find_one_and_update(
            {"_id": project_id, "owner_id": owner_id},
            {"$set": changes},
            return_document=ReturnDocument.AFTER,
        )

    async def delete_for_owner(self, project_id: str, owner_id: str) -> bool:
        result = await self._collection.delete_one({"_id": project_id, "owner_id": owner_id})
        return result.deleted_count == 1
