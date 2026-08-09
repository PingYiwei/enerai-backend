from __future__ import annotations

from datetime import UTC, datetime

from pymongo.errors import DuplicateKeyError

from app.core.config import Settings
from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal, create_access_token, hash_password, verify_password
from app.modules.auth.repository import Document, UserRepository
from app.modules.auth.schemas import RegisterRequest, TokenResponse, UserResponse


def user_response(document: Document) -> UserResponse:
    return UserResponse(
        id=document["_id"],
        username=document["username"],
        email=document["email"],
        created_at=document["created_at"],
    )


async def register_user(repository: UserRepository, request: RegisterRequest) -> UserResponse:
    now = datetime.now(UTC)
    document = {
        "_id": new_id("usr"),
        "username": request.username.strip(),
        "username_key": request.username.strip().casefold(),
        "email": str(request.email).strip().casefold(),
        "password_hash": hash_password(request.password),
        "created_at": now,
        "updated_at": now,
    }
    try:
        await repository.insert(document)
    except DuplicateKeyError as error:
        raise AppError(
            "identity_already_exists",
            "Username or email is already registered",
            status_code=409,
        ) from error
    return user_response(document)


async def authenticate_user(
    repository: UserRepository,
    username: str,
    password: str,
    settings: Settings,
) -> TokenResponse:
    document = await repository.find_by_username_key(username.strip().casefold())
    if document is None or not verify_password(password, document["password_hash"]):
        raise AppError("invalid_credentials", "Username or password is incorrect", status_code=401)

    user = user_response(document)
    token, expires_at = create_access_token(
        Principal(user_id=user.id, username=user.username),
        settings,
    )
    return TokenResponse(access_token=token, expires_at=expires_at, user=user)
