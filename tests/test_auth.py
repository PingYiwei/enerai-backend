from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import Principal, create_access_token, decode_access_token
from app.modules.auth.schemas import RegisterRequest
from app.modules.auth.service import authenticate_user, register_user
from tests.fakes import InMemoryUserRepository


async def test_register_and_authenticate_user() -> None:
    repository = InMemoryUserRepository()
    settings = Settings(environment="test", jwt_secret="test-secret-with-at-least-32-bytes")

    user = await register_user(
        repository,
        RegisterRequest(username="Ada", email="ada@example.com", password="correct-horse"),
    )
    token = await authenticate_user(repository, "ada", "correct-horse", settings)

    assert token.user == user
    assert decode_access_token(token.access_token, settings) == Principal(
        user_id=user.id,
        username="Ada",
    )


async def test_wrong_password_is_rejected() -> None:
    repository = InMemoryUserRepository()
    settings = Settings(environment="test", jwt_secret="test-secret-with-at-least-32-bytes")
    await register_user(
        repository,
        RegisterRequest(username="Ada", email="ada@example.com", password="correct-horse"),
    )

    with pytest.raises(AppError, match="Username or password") as raised:
        await authenticate_user(repository, "Ada", "wrong-password", settings)

    assert raised.value.code == "invalid_credentials"


def test_access_token_round_trip() -> None:
    settings = Settings(environment="test", jwt_secret="test-secret-with-at-least-32-bytes")
    principal = Principal(user_id="usr_test", username="test")
    token, _ = create_access_token(principal, settings)
    assert decode_access_token(token, settings) == principal
