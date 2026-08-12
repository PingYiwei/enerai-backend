from __future__ import annotations

import re

from app.modules.agents.providers.base import Provider
from app.modules.agents.runtime.types import Message, ProviderRequest, TextDelta

_TITLE_PREFIX = re.compile(r"^(?:session\s+)?title\s*:\s*", re.IGNORECASE)


async def generate_session_title(provider: Provider, model: str, user_message: str) -> str:
    chunks: list[str] = []
    request = ProviderRequest(
        model=model,
        system_prompt=(
            "Generate one concise title for an EnerAI Insight session from the user's first "
            "message. Match the user's language. Return only the title, without quotes, markdown, "
            "labels, or explanation. Keep it under 40 characters."
        ),
        messages=(Message(role="user", content=user_message[:4000]),),
        temperature=0.2,
        max_output_tokens=48,
    )
    async for event in provider.stream(request):
        if isinstance(event, TextDelta):
            chunks.append(event.delta)
    return normalize_session_title("".join(chunks))


def normalize_session_title(value: str) -> str:
    line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    line = _TITLE_PREFIX.sub("", line).strip().strip("`#* ")
    line = line.strip("\"'“”‘’").strip()
    line = line.rstrip(".。!！?？:：").strip()
    if len(line) > 60:
        line = line[:60].rstrip()
    return line
