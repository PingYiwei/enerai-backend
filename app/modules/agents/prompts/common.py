from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo


def render_runtime_context(
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Shanghai",
) -> str:
    instant = now or datetime.now(UTC)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=UTC)
    instant = instant.astimezone(UTC)
    local = instant.astimezone(ZoneInfo(timezone_name))
    return "\n".join(
        (
            "Runtime context (authoritative for this run):",
            f"- Current UTC time: {instant.isoformat(timespec='seconds')}",
            f"- Current local time: {local.isoformat(timespec='seconds')}",
            f"- Local timezone: {timezone_name}",
            f"- Local calendar date: {local.date().isoformat()} ({local.strftime('%A')})",
            "- Interpret relative dates such as today, yesterday, and last week using the local "
            "timezone above.",
            "- The clock is context only. Use project tools for mutable project state and measured "
            "operational data.",
        )
    )
