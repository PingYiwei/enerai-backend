from __future__ import annotations

from typing import Literal

from pydantic import SecretStr

from app.core.config import Settings
from app.core.errors import AppError
from app.modules.agents.providers.openai_compatible import (
    ApiStyle,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)

ProviderId = Literal["openai", "openrouter", "bailian"]


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[tuple[ProviderId, ApiStyle], OpenAICompatibleProvider] = {}

    def get(
        self,
        provider_id: ProviderId,
        api_style: ApiStyle | None = None,
    ) -> OpenAICompatibleProvider:
        configured_style, base_url, secret = self._configuration(provider_id)
        resolved_style = api_style or configured_style
        key = (provider_id, resolved_style)
        existing = self._providers.get(key)
        if existing is not None:
            return existing
        if secret is None or not secret.get_secret_value():
            raise AppError(
                "provider_not_configured",
                f"Provider {provider_id} is not configured",
                status_code=503,
            )
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                id=provider_id,
                api_style=resolved_style,
                base_url=base_url,
                api_key=secret.get_secret_value(),
                headers=(
                    {"HTTP-Referer": "https://nodex.ai", "X-Title": "Nodex"}
                    if provider_id == "openrouter"
                    else {}
                ),
            )
        )
        self._providers[key] = provider
        return provider

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()

    def _configuration(self, provider_id: ProviderId) -> tuple[ApiStyle, str, SecretStr | None]:
        if provider_id == "openai":
            return (
                self._settings.openai_api_style,
                self._settings.openai_base_url,
                self._settings.openai_api_key,
            )
        if provider_id == "openrouter":
            return (
                self._settings.openrouter_api_style,
                self._settings.openrouter_base_url,
                self._settings.openrouter_api_key,
            )
        return (
            self._settings.bailian_api_style,
            self._settings.bailian_base_url,
            self._settings.bailian_api_key,
        )
