from __future__ import annotations

from collections.abc import AsyncIterator

from app.modules.agents.runtime.titles import generate_session_title, normalize_session_title
from app.modules.agents.runtime.types import (
    ProviderEvent,
    ProviderRequest,
    ResponseCompleted,
    TextDelta,
)


class TitleProvider:
    def __init__(self) -> None:
        self.request: ProviderRequest | None = None

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        self.request = request
        yield TextDelta('Title: "Cooling Plant ')
        yield TextDelta('Efficiency Review"')
        yield ResponseCompleted("stop")


async def test_generate_session_title_uses_small_independent_request() -> None:
    provider = TitleProvider()

    title = await generate_session_title(provider, "aux-model", "Analyze this cooling plant")

    assert title == "Cooling Plant Efficiency Review"
    assert provider.request is not None
    assert provider.request.model == "aux-model"
    assert provider.request.tools == ()
    assert provider.request.max_output_tokens == 48


def test_normalize_session_title_limits_and_cleans_output() -> None:
    assert normalize_session_title("# 能源站效率分析。\nextra") == "能源站效率分析"
    assert len(normalize_session_title("x" * 100)) == 60
