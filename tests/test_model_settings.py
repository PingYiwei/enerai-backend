from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.security import encrypt_provider_secret
from app.modules.auth.model_settings import read_model_settings, resolve_provider_runtime


class FakeCollection:
    def __init__(self, document: dict[str, Any]) -> None:
        self.document = document

    async def find_one(self, _: dict[str, Any]) -> dict[str, Any]:
        return self.document


class FakeDatabase:
    def __init__(self, document: dict[str, Any]) -> None:
        self.user_model_settings = FakeCollection(document)


@pytest.mark.asyncio
async def test_user_provider_settings_are_masked_and_drive_multimodal_runs() -> None:
    settings = Settings(
        provider_secret_key=SecretStr("MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=")
    )
    encrypted = encrypt_provider_secret("sk-user-provider", settings)
    database = FakeDatabase(
        {
            "_id": "usr_test",
            "active_provider": "openrouter",
            "providers": {
                "openrouter": {
                    "api_key_encrypted": encrypted,
                    "api_key_last_four": "ider",
                    "models": {
                        "primary": "primary-model",
                        "text": "text-model",
                        "multimodal": "vision-model",
                        "auxiliary": "fast-model",
                    },
                }
            },
        }
    )

    response = await read_model_settings(database, "usr_test", settings)  # type: ignore[arg-type]
    runtime = await resolve_provider_runtime(  # type: ignore[arg-type]
        database,
        "usr_test",
        settings,
        requested_provider=None,
        requested_api_style=None,
        requested_model=None,
        multimodal=True,
    )

    assert response.providers["openrouter"].api_key_masked == "••••••••ider"
    assert encrypted not in response.model_dump_json()
    assert runtime.provider == "openrouter"
    assert runtime.api_key == "sk-user-provider"
    assert runtime.model == "vision-model"
