from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from pymongo.asynchronous.database import AsyncDatabase

from app.modules.agents.runtime.types import (
    Message,
    ProviderRequest,
    ReasoningDelta,
    ResponseCompleted,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    TraceContext,
    Usage,
    UsageUpdated,
)
from app.modules.traces.recorder import TraceRecorder

Document = dict[str, Any]


class Collection:
    def __init__(self, documents: list[Document] | None = None) -> None:
        self.documents = documents or []

    async def insert_one(self, document: Document) -> None:
        self.documents.append(deepcopy(document))

    async def find_one(self, query: Document) -> Document | None:
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return deepcopy(document)
        return None

    async def update_one(self, query: Document, update: Document) -> None:
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                document.update(deepcopy(update["$set"]))
                return


class Database:
    def __init__(self, pricing: list[Document] | None = None) -> None:
        self.llm_traces = Collection()
        self.trace_model_pricing = Collection(pricing)


async def test_trace_recorder_captures_stream_and_calculates_snapshot_cost() -> None:
    database = Database(
        [
            {
                "provider": "openai",
                "model": "test-model",
                "input_per_million_usd": "2",
                "output_per_million_usd": "4",
                "cached_input_per_million_usd": "1",
                "reasoning_per_million_usd": "6",
                "effective_from": "2026-01-01",
            }
        ]
    )
    recorder = TraceRecorder(cast(AsyncDatabase[Document], database))
    session = await recorder.start(
        "openai",
        "responses",
        ProviderRequest(
            model="test-model",
            system_prompt="Be precise",
            messages=(Message(role="user", content="hello"),),
            trace=TraceContext(
                user_id="usr_1", source="insight", run_id="run_1", turn=2
            ),
        ),
    )
    assert session is not None
    session.observe(TextDelta("answer"))
    session.observe(ReasoningDelta("reason"))
    session.observe(ToolCallStarted(index=0, id="call_1", name="lookup"))
    session.observe(ToolCallArgumentsDelta(index=0, delta='{"id":1}'))
    session.observe(
        UsageUpdated(
            Usage(
                input_tokens=100,
                output_tokens=20,
                cached_input_tokens=10,
                reasoning_tokens=5,
            )
        )
    )
    session.observe(ResponseCompleted(stop_reason="tool_calls", response_id="resp_1"))
    await session.finish("success")

    trace = database.llm_traces.documents[0]
    assert trace["status"] == "success"
    assert trace["source"] == "insight"
    assert trace["turn"] == 2
    assert trace["response"]["output_text"] == "answer"
    assert trace["response"]["reasoning_text"] == "reason"
    assert trace["response"]["tool_calls"][0]["arguments"] == '{"id":1}'
    assert trace["usage"]["total_tokens"] == 120
    assert trace["cost"]["priced"] is True
    assert trace["cost"]["total_usd_micros"] == 280
    assert trace["provider_response_id"] == "resp_1"


async def test_trace_recorder_preserves_error_when_model_is_unpriced() -> None:
    database = Database()
    recorder = TraceRecorder(cast(AsyncDatabase[Document], database))
    session = await recorder.start(
        "openrouter",
        "chat_completions",
        ProviderRequest(
            model="unknown",
            messages=(Message(role="user", content="hello"),),
            trace=TraceContext(user_id="usr_1", source="optimizer"),
        ),
    )
    assert session is not None
    error = RuntimeError("provider unavailable")
    await session.finish("error", error)

    trace = database.llm_traces.documents[0]
    assert trace["status"] == "error"
    assert trace["error"] == {"type": "RuntimeError", "message": "provider unavailable"}
    assert trace["cost"]["priced"] is False
    assert trace["cost"]["total_usd_micros"] is None
