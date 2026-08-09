from __future__ import annotations

from hashlib import sha256
from typing import Literal

from app.core.errors import AppError
from app.modules.agents.providers.openai_compatible import (
    ApiStyle,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)

ProviderId = Literal["openai", "openrouter", "bailian"]


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[
            tuple[ProviderId, ApiStyle, str, str], OpenAICompatibleProvider
        ] = {}

    def get(
        self,
        provider_id: ProviderId,
        api_style: ApiStyle,
        *,
        api_key: str,
        base_url: str,
    ) -> OpenAICompatibleProvider:
        key = (
            provider_id,
            api_style,
            base_url,
            sha256(api_key.encode()).hexdigest(),
        )
        existing = self._providers.get(key)
        if existing is not None:
            return existing
        if not api_key:
            raise AppError(
                "provider_not_configured",
                f"Provider {provider_id} is not configured",
                status_code=503,
            )
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                id=provider_id,
                api_style=api_style,
                base_url=base_url,
                api_key=api_key,
                headers=(
                    {"HTTP-Referer": "https://enerai.ai", "X-Title": "EnerAI"}
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
