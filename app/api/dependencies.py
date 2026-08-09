from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, cast

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.security import Principal, decode_access_token, hash_api_key
from app.modules.auth.repository import MongoUserRepository, UserRepository
from app.modules.projects.repository import MongoProjectRepository, ProjectRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


Document = dict[str, Any]


def get_database(request: Request) -> AsyncDatabase[Document]:
    return cast(AsyncDatabase[Document], request.app.state.database)


async def get_principal(
    token: Annotated[str, Depends(oauth2_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
    database: Annotated[AsyncDatabase[Document], Depends(get_database)],
) -> Principal:
    if not token.startswith("ndx_"):
        return decode_access_token(token, settings)
    now = datetime.now(UTC)
    document = await database.user_api_keys.find_one_and_update(
        {"key_hash": hash_api_key(token), "revoked_at": None},
        {"$set": {"last_used_at": now}},
    )
    if document is None:
        raise AppError("invalid_api_key", "API key is invalid or revoked", status_code=401)
    return Principal(user_id=document["owner_id"], username=document["username"])


def get_user_repository(database: Database) -> UserRepository:
    return MongoUserRepository(database)


def get_project_repository(database: Database) -> ProjectRepository:
    return MongoProjectRepository(database)


Database = Annotated[AsyncDatabase[Document], Depends(get_database)]
CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
Users = Annotated[UserRepository, Depends(get_user_repository)]
Projects = Annotated[ProjectRepository, Depends(get_project_repository)]
