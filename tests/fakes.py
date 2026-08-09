from __future__ import annotations

from copy import deepcopy

from pymongo.errors import DuplicateKeyError

from app.modules.auth.repository import Document as UserDocument
from app.modules.projects.repository import Document as ProjectDocument


class InMemoryUserRepository:
    def __init__(self) -> None:
        self.documents: dict[str, UserDocument] = {}

    async def insert(self, document: UserDocument) -> None:
        if any(
            existing["username_key"] == document["username_key"]
            or existing["email"] == document["email"]
            for existing in self.documents.values()
        ):
            raise DuplicateKeyError("duplicate identity")
        self.documents[document["_id"]] = deepcopy(document)

    async def find_by_username_key(self, username_key: str) -> UserDocument | None:
        for document in self.documents.values():
            if document["username_key"] == username_key:
                return deepcopy(document)
        return None


class InMemoryProjectRepository:
    def __init__(self) -> None:
        self.documents: dict[str, ProjectDocument] = {}

    async def insert(self, document: ProjectDocument) -> None:
        if any(
            existing["owner_id"] == document["owner_id"]
            and existing["name_key"] == document["name_key"]
            for existing in self.documents.values()
        ):
            raise DuplicateKeyError("duplicate project name")
        self.documents[document["_id"]] = deepcopy(document)

    async def list_for_owner(self, owner_id: str) -> list[ProjectDocument]:
        documents = [
            deepcopy(document)
            for document in self.documents.values()
            if document["owner_id"] == owner_id
        ]
        return sorted(documents, key=lambda document: document["updated_at"], reverse=True)

    async def get_for_owner(self, project_id: str, owner_id: str) -> ProjectDocument | None:
        document = self.documents.get(project_id)
        if document is None or document["owner_id"] != owner_id:
            return None
        return deepcopy(document)

    async def update_for_owner(
        self,
        project_id: str,
        owner_id: str,
        changes: ProjectDocument,
    ) -> ProjectDocument | None:
        document = self.documents.get(project_id)
        if document is None or document["owner_id"] != owner_id:
            return None
        if "name_key" in changes and any(
            other_id != project_id
            and other["owner_id"] == owner_id
            and other["name_key"] == changes["name_key"]
            for other_id, other in self.documents.items()
        ):
            raise DuplicateKeyError("duplicate project name")
        document.update(deepcopy(changes))
        return deepcopy(document)

    async def delete_for_owner(self, project_id: str, owner_id: str) -> bool:
        document = self.documents.get(project_id)
        if document is None or document["owner_id"] != owner_id:
            return False
        del self.documents[project_id]
        return True
