from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import decrypt_provider_secret, encrypt_provider_secret
from app.modules.auth.schemas import (
    ModelRoles,
    ModelSettingsResponse,
    ModelSettingsUpdate,
    ProviderId,
    ProviderModelConfiguration,
)

Document = dict[str, Any]
ApiStyle = Literal["responses", "chat_completions"]
PROVIDERS: tuple[ProviderId, ...] = ("openai", "openrouter", "bailian")


@dataclass(frozen=True, slots=True)
class ProviderRuntimeConfiguration:
    provider: ProviderId
    api_style: ApiStyle
    base_url: str
    api_key: str
    model: str


def provider_base_url(settings: Settings, provider: ProviderId) -> str:
    if provider == "openai":
        return settings.openai_base_url
    if provider == "openrouter":
        return settings.openrouter_base_url
    return settings.bailian_base_url


async def read_model_settings(
    database: AsyncDatabase[Document], owner_id: str, settings: Settings
) -> ModelSettingsResponse:
    document = await database.user_model_settings.find_one({"_id": owner_id})
    saved_providers = _saved_providers(document)
    configurations: dict[ProviderId, ProviderModelConfiguration] = {}
    for provider in PROVIDERS:
        api_style: ApiStyle = "responses"
        base_url = provider_base_url(settings, provider)
        saved = saved_providers.get(provider, {})
        saved_api_style = saved.get("api_style")
        if saved_api_style in {"responses", "chat_completions"}:
            api_style = cast(ApiStyle, saved_api_style)
        last_four = str(saved.get("api_key_last_four") or "")
        configured = bool(saved.get("api_key_encrypted"))
        configurations[provider] = ProviderModelConfiguration(
            provider=provider,
            api_style=api_style,
            base_url=base_url,
            models=ModelRoles.model_validate(saved.get("models") or {}),
            api_key_configured=configured,
            api_key_masked=f"••••••••{last_four}" if configured and last_four else None,
        )
    active = (
        str(document.get("active_provider"))
        if document is not None and document.get("active_provider") in PROVIDERS
        else settings.default_provider
    )
    return ModelSettingsResponse(
        active_provider=cast(ProviderId, active),
        providers=configurations,
        updated_at=document.get("updated_at") if document else None,
    )


async def write_model_settings(
    database: AsyncDatabase[Document],
    owner_id: str,
    body: ModelSettingsUpdate,
    settings: Settings,
) -> ModelSettingsResponse:
    now = datetime.now(UTC)
    provider_path = f"providers.{body.provider}"
    update: Document = {
        "$set": {
            "owner_id": owner_id,
            "active_provider": body.provider,
            f"{provider_path}.api_style": body.api_style,
            f"{provider_path}.models": body.models.model_dump(),
            "updated_at": now,
        },
        "$setOnInsert": {"created_at": now},
    }
    if body.api_key is not None:
        update["$set"][f"{provider_path}.api_key_encrypted"] = encrypt_provider_secret(
            body.api_key, settings
        )
        update["$set"][f"{provider_path}.api_key_last_four"] = body.api_key[-4:]
    elif body.clear_api_key:
        update["$unset"] = {
            f"{provider_path}.api_key_encrypted": "",
            f"{provider_path}.api_key_last_four": "",
        }
    await database.user_model_settings.update_one({"_id": owner_id}, update, upsert=True)
    return await read_model_settings(database, owner_id, settings)


async def resolve_provider_runtime(
    database: AsyncDatabase[Document],
    owner_id: str,
    settings: Settings,
    *,
    requested_provider: ProviderId | None,
    requested_api_style: ApiStyle | None,
    requested_model: str | None,
    multimodal: bool,
) -> ProviderRuntimeConfiguration:
    document = await database.user_model_settings.find_one({"_id": owner_id})
    saved_providers = _saved_providers(document)
    active = document.get("active_provider") if document else None
    provider = requested_provider or (
        cast(ProviderId, active) if active in PROVIDERS else settings.default_provider
    )
    saved = saved_providers.get(provider, {})
    models = ModelRoles.model_validate(saved.get("models") or {})
    fallback_models = (
        (models.multimodal, models.primary, models.text)
        if multimodal
        else (models.primary, models.text)
    )
    model = requested_model or next((item for item in fallback_models if item), "")
    if not model:
        raise AppError(
            "model_not_configured",
            "Configure a model for this provider before starting an Agent run",
            status_code=422,
        )

    api_style: ApiStyle = "responses"
    base_url = provider_base_url(settings, provider)
    saved_api_style = saved.get("api_style")
    if saved_api_style in {"responses", "chat_completions"}:
        api_style = cast(ApiStyle, saved_api_style)
    encrypted = saved.get("api_key_encrypted")
    if not isinstance(encrypted, str) or not encrypted:
        raise AppError(
            "provider_not_configured",
            f"Configure an API key for {provider} before starting an Agent run",
            status_code=422,
        )
    api_key = decrypt_provider_secret(encrypted, settings)
    return ProviderRuntimeConfiguration(
        provider=provider,
        api_style=requested_api_style or api_style,
        base_url=base_url,
        api_key=api_key,
        model=model.strip(),
    )


def _saved_providers(document: Document | None) -> dict[str, Document]:
    if document is None or not isinstance(document.get("providers"), dict):
        return {}
    return {
        str(key): value for key, value in document["providers"].items() if isinstance(value, dict)
    }
