from __future__ import annotations

from datetime import datetime

from app.modules.agents.prompts.assignment import ASSIGNMENT_SYSTEM_PROMPT
from app.modules.agents.prompts.common import render_runtime_context
from app.modules.agents.prompts.insight import INSIGHT_SYSTEM_PROMPT
from app.modules.agents.prompts.inspection import INSPECTION_SYSTEM_PROMPT
from app.modules.agents.prompts.studio import STUDIO_SYSTEM_PROMPT


def render_agent_system_prompt(
    surface: str,
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Shanghai",
) -> str:
    prompt = (
        STUDIO_SYSTEM_PROMPT
        if surface == "studio"
        else ASSIGNMENT_SYSTEM_PROMPT
        if surface == "assignment"
        else INSPECTION_SYSTEM_PROMPT
        if surface == "inspection"
        else INSIGHT_SYSTEM_PROMPT
    )
    runtime_context = render_runtime_context(now=now, timezone_name=timezone_name)
    return f"{prompt}\n\n{runtime_context}"


__all__ = [
    "ASSIGNMENT_SYSTEM_PROMPT",
    "INSIGHT_SYSTEM_PROMPT",
    "INSPECTION_SYSTEM_PROMPT",
    "STUDIO_SYSTEM_PROMPT",
    "render_agent_system_prompt",
]
