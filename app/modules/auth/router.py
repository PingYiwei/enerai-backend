from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pymongo import DESCENDING

from app.api.dependencies import CurrentPrincipal, Database, Users
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import create_api_key
from app.modules.auth.schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyList,
    ApiKeySummary,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.modules.auth.service import authenticate_user, register_user

router = APIRouter()
Document = dict[str, Any]


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest, users: Users) -> UserResponse:
    return await register_user(users, request)


@router.post("/token", response_model=TokenResponse)
async def token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    users: Users,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenResponse:
    return await authenticate_user(users, form.username, form.password, settings)


@router.get("/me", response_model=UserResponse)
async def me(database: Database, principal: CurrentPrincipal) -> UserResponse:
    document = await database.users.find_one({"_id": principal.user_id})
    if document is None:
        raise AppError("user_not_found", "User was not found", status_code=404)
    return UserResponse(
        id=document["_id"],
        username=document["username"],
        email=document["email"],
        created_at=document["created_at"],
    )


def _key_summary(document: Document) -> ApiKeySummary:
    return ApiKeySummary(
        id=str(document["_id"]),
        name=str(document["name"]),
        prefix=str(document["prefix"]),
        created_at=document["created_at"],
        last_used_at=document.get("last_used_at"),
    )


@router.get("/api-keys", response_model=ApiKeyList)
async def api_keys(database: Database, principal: CurrentPrincipal) -> ApiKeyList:
    documents = await (
        database.user_api_keys.find({"owner_id": principal.user_id, "revoked_at": None})
        .sort("created_at", DESCENDING)
        .to_list(None)
    )
    return ApiKeyList(
        items=[_key_summary(document) for document in documents], total=len(documents)
    )


@router.post("/api-keys", response_model=ApiKeyCreated, status_code=status.HTTP_201_CREATED)
async def create_key(
    body: ApiKeyCreate, database: Database, principal: CurrentPrincipal
) -> ApiKeyCreated:
    secret, key_hash = create_api_key()
    now = datetime.now(UTC)
    document = {
        "_id": new_id("key"),
        "owner_id": principal.user_id,
        "username": principal.username,
        "name": body.name.strip(),
        "prefix": f"{secret[:12]}…{secret[-4:]}",
        "key_hash": key_hash,
        "created_at": now,
        "last_used_at": None,
        "revoked_at": None,
    }
    await database.user_api_keys.insert_one(document)
    return ApiKeyCreated(**_key_summary(document).model_dump(), secret=secret)


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(key_id: str, database: Database, principal: CurrentPrincipal) -> Response:
    result = await database.user_api_keys.update_one(
        {"_id": key_id, "owner_id": principal.user_id, "revoked_at": None},
        {"$set": {"revoked_at": datetime.now(UTC)}},
    )
    if not result.modified_count:
        raise AppError("api_key_not_found", "API key was not found", status_code=404)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
