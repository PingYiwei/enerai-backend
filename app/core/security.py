from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

import jwt
from pwdlib import PasswordHash

from app.core.config import Settings
from app.core.errors import AppError

_password_hash = PasswordHash.recommended()


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    username: str


def hash_password(password: str) -> str:
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _password_hash.verify(password, password_hash)


def create_access_token(principal: Principal, settings: Settings) -> tuple[str, datetime]:
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_minutes)
    payload: dict[str, Any] = {
        "sub": principal.user_id,
        "username": principal.username,
        "iat": datetime.now(UTC),
        "exp": expires_at,
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_at


def decode_access_token(token: str, settings: Settings) -> Principal:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id = str(payload["sub"])
        username = str(payload["username"])
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as error:
        raise AppError(
            "invalid_access_token",
            "Access token is invalid or expired",
            status_code=401,
        ) from error
    return Principal(user_id=user_id, username=username)


def create_api_key() -> tuple[str, str]:
    value = f"ndx_{secrets.token_urlsafe(32)}"
    return value, hash_api_key(value)


def hash_api_key(value: str) -> str:
    return sha256(value.encode()).hexdigest()
