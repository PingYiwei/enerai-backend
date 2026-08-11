from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.modules.agents.runtime.types import ProviderEvent, ProviderRequest


class Provider(Protocol):
    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]: ...
