from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

import jwt
from cryptography.fernet import Fernet, InvalidToken
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


def encrypt_provider_secret(value: str, settings: Settings) -> str:
    """Encrypt a recoverable provider credential before it is persisted."""
    return _provider_cipher(settings).encrypt(value.encode()).decode()


def decrypt_provider_secret(value: str, settings: Settings) -> str:
    try:
        return _provider_cipher(settings).decrypt(value.encode()).decode()
    except InvalidToken as error:
        raise AppError(
            "provider_secret_unreadable",
            "The saved provider credential cannot be decrypted with this server key",
            status_code=503,
        ) from error


def _provider_cipher(settings: Settings) -> Fernet:
    secret = settings.provider_secret_key
    if secret is None or not secret.get_secret_value():
        raise AppError(
            "provider_secret_key_missing",
            "Provider credential encryption is not configured on this server",
            status_code=503,
        )
    try:
        return Fernet(secret.get_secret_value().encode())
    except (TypeError, ValueError) as error:
        raise AppError(
            "provider_secret_key_invalid",
            "Provider credential encryption key is invalid",
            status_code=503,
        ) from error
