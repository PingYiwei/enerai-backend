from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from time import monotonic
from typing import Any, Literal

from pymongo.asynchronous.database import AsyncDatabase

from app.core.ids import new_id
from app.modules.agents.runtime.types import (
    ProviderEvent,
    ProviderRequest,
    ReasoningDelta,
    ResponseCompleted,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    Usage,
    UsageUpdated,
)

logger = logging.getLogger(__name__)
Document = dict[str, Any]
TraceStatus = Literal["success", "error", "cancelled"]


class TraceRecorder:
    """Writes provider traces without making observability a core-path dependency."""

    def __init__(self, database: AsyncDatabase[Document]) -> None:
        self._database = database

    async def start(
        self,
        provider: str,
        api_style: str,
        request: ProviderRequest,
    ) -> TraceSession | None:
        context = request.trace
        if context is None:
            return None
        session = TraceSession(self._database, provider, api_style, request)
        try:
            await session.start()
        except Exception:
            logger.exception("Failed to start LLM trace")
            return None
        return session


class TraceSession:
    def __init__(
        self,
        database: AsyncDatabase[Document],
        provider: str,
        api_style: str,
        request: ProviderRequest,
    ) -> None:
        self._database = database
        self._provider = provider
        self._api_style = api_style
        self._request = request
        self._id = new_id("trc")
        self._started_at = datetime.now(UTC)
        self._started_clock = monotonic()
        self._first_token_at: datetime | None = None
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage = Usage()
        self._response_id: str | None = None
        self._stop_reason: str | None = None
        self._events: list[Document] = []

    async def start(self) -> None:
        context = self._request.trace
        assert context is not None
        await self._database.llm_traces.insert_one(
            {
                "_id": self._id,
                "provider_trace_id": None,
                "provider_response_id": None,
                "owner_id": context.user_id,
                "username": context.username,
                "source": context.source,
                "feature": context.feature,
                "project_id": context.project_id,
                "session_id": context.session_id,
                "run_id": context.run_id,
                "turn": context.turn,
                "parent_trace_id": context.parent_trace_id,
                "tags": list(context.tags),
                "provider": self._provider,
                "api_style": self._api_style,
                "model": self._request.model,
                "temperature": self._request.temperature,
                "max_output_tokens": self._request.max_output_tokens,
                "request": {
                    "system_prompt": self._request.system_prompt,
                    "messages": [_message_document(message) for message in self._request.messages],
                    "tools": [asdict(tool) for tool in self._request.tools],
                },
                "response": None,
                "status": "running",
                "error": None,
                "usage": _usage_document(Usage()),
                "cost": _unpriced_cost(),
                "started_at": self._started_at,
                "first_token_at": None,
                "completed_at": None,
                "duration_ms": None,
                "time_to_first_token_ms": None,
                "created_at": self._started_at,
                "updated_at": self._started_at,
            }
        )

    def observe(self, event: ProviderEvent) -> None:
        offset_ms = round((monotonic() - self._started_clock) * 1000)
        if isinstance(event, (TextDelta, ReasoningDelta)) and self._first_token_at is None:
            self._first_token_at = datetime.now(UTC)
        if isinstance(event, TextDelta):
            self._text.append(event.delta)
            return
        if isinstance(event, ReasoningDelta):
            self._reasoning.append(event.delta)
            return
        if isinstance(event, ToolCallStarted):
            self._tool_calls[event.index] = {
                "id": event.id,
                "name": event.name,
                "arguments": "",
            }
            self._events.append(
                {"type": "tool_call_started", "offset_ms": offset_ms, "name": event.name}
            )
            return
        if isinstance(event, ToolCallArgumentsDelta):
            call = self._tool_calls.setdefault(
                event.index, {"id": "", "name": "", "arguments": ""}
            )
            call["arguments"] += event.delta
            return
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            self._events.append(
                {"type": "usage", "offset_ms": offset_ms, **_usage_document(event.usage)}
            )
            return
        if isinstance(event, ResponseCompleted):
            self._response_id = event.response_id
            self._stop_reason = event.stop_reason
            self._events.append(
                {
                    "type": "response_completed",
                    "offset_ms": offset_ms,
                    "stop_reason": event.stop_reason,
                }
            )

    async def finish(
        self,
        status: TraceStatus,
        error: BaseException | None = None,
    ) -> None:
        try:
            await self._finish(status, error)
        except Exception:
            logger.exception("Failed to finish LLM trace %s", self._id)

    async def _finish(self, status: TraceStatus, error: BaseException | None) -> None:
        completed_at = datetime.now(UTC)
        pricing = await self._database.trace_model_pricing.find_one(
            {"provider": self._provider, "model": self._request.model}
        )
        cost = _calculate_cost(self._usage, pricing)
        duration_ms = round((monotonic() - self._started_clock) * 1000)
        first_token_ms = (
            round((self._first_token_at - self._started_at).total_seconds() * 1000)
            if self._first_token_at is not None
            else None
        )
        await self._database.llm_traces.update_one(
            {"_id": self._id},
            {
                "$set": {
                    "provider_response_id": self._response_id,
                    "response": {
                        "output_text": "".join(self._text),
                        "reasoning_text": "".join(self._reasoning),
                        "tool_calls": [self._tool_calls[key] for key in sorted(self._tool_calls)],
                        "stop_reason": self._stop_reason,
                        "events": self._events,
                    },
                    "status": status,
                    "error": (
                        {"type": type(error).__name__, "message": str(error)}
                        if error is not None
                        else None
                    ),
                    "usage": _usage_document(self._usage),
                    "cost": cost,
                    "first_token_at": self._first_token_at,
                    "completed_at": completed_at,
                    "duration_ms": duration_ms,
                    "time_to_first_token_ms": first_token_ms,
                    "updated_at": completed_at,
                }
            },
        )


def _message_document(message: Any) -> Document:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [asdict(call) for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "images": [asdict(image) for image in message.images],
    }


def _usage_document(usage: Usage) -> Document:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }


def _unpriced_cost() -> Document:
    return {
        "priced": False,
        "currency": "USD",
        "input_usd_micros": None,
        "output_usd_micros": None,
        "cached_input_usd_micros": None,
        "reasoning_usd_micros": None,
        "total_usd_micros": None,
        "pricing_snapshot": None,
    }


def _calculate_cost(usage: Usage, pricing: Document | None) -> Document:
    if pricing is None:
        return _unpriced_cost()
    rates = {
        key: Decimal(str(pricing.get(f"{key}_per_million_usd", "0")))
        for key in ("input", "output", "cached_input", "reasoning")
    }
    billable_input = max(usage.input_tokens - usage.cached_input_tokens, 0)
    billable_output = max(usage.output_tokens - usage.reasoning_tokens, 0)
    token_counts = {
        "input": billable_input,
        "output": billable_output,
        "cached_input": usage.cached_input_tokens,
        "reasoning": usage.reasoning_tokens,
    }
    components = {
        key: int((Decimal(count) * rates[key]).quantize(Decimal("1"), ROUND_HALF_UP))
        for key, count in token_counts.items()
    }
    return {
        "priced": True,
        "currency": "USD",
        **{f"{key}_usd_micros": value for key, value in components.items()},
        "total_usd_micros": sum(components.values()),
        "pricing_snapshot": {
            **{f"{key}_per_million_usd": str(value) for key, value in rates.items()},
            "effective_from": pricing.get("effective_from"),
        },
    }
