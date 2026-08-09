from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.modules.agents.engine import AgentEngine, AgentRunRequest, select_provider_context
from app.modules.agents.tools import Tool, ToolContext, execute_tool_batch
from app.modules.agents.types import (
    JsonObject,
    Message,
    ProviderEvent,
    ProviderRequest,
    ResponseCompleted,
    TextDelta,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    ToolResult,
    Usage,
    UsageUpdated,
)


class ScriptedProvider:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield TextDelta("Checking. ")
            yield ToolCallStarted(index=0, id="call_1", name="read_value")
            yield ToolCallArgumentsDelta(index=0, delta='{"name":"temperature"}')
            yield UsageUpdated(Usage(input_tokens=10, output_tokens=4))
            yield ResponseCompleted("tool_calls", "resp_1")
            return
        assert request.messages[-1] == Message(
            role="tool",
            content="temperature=7.2",
            tool_call_id="call_1",
        )
        yield TextDelta("Temperature is 7.2 °C.")
        yield UsageUpdated(Usage(input_tokens=18, output_tokens=7))
        yield ResponseCompleted("stop", "resp_2")


async def test_agent_runs_tool_turn_then_final_turn() -> None:
    provider = ScriptedProvider()
    events: list[tuple[str, JsonObject]] = []

    async def execute(arguments: JsonObject, context: ToolContext) -> ToolResult:
        assert arguments == {"name": "temperature"}
        assert context.project_id == "prj_1"
        return ToolResult(tool_call_id="call_1", content="temperature=7.2")

    async def emit(event_type: str, data: JsonObject) -> None:
        events.append((event_type, data))

    result = await AgentEngine(provider).run(
        AgentRunRequest(
            run_id="run_1",
            session_id="ses_1",
            project_id="prj_1",
            user_id="usr_1",
            model="test-model",
            system_prompt="Be precise.",
            messages=(Message(role="user", content="Read the temperature"),),
            tools=(
                Tool(
                    name="read_value",
                    description="Read one value",
                    input_schema={
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                    execute=execute,
                ),
            ),
        ),
        emit,
    )

    assert result.final_message.content == "Temperature is 7.2 °C."
    assert result.turns == 2
    assert result.usage == Usage(input_tokens=28, output_tokens=11)
    assert [event_type for event_type, _ in events].count("tool_execution_start") == 1
    assert events[-1] == ("run_end", {"run_id": "run_1", "outcome": "completed"})


async def test_parallel_tools_emit_completion_order_but_return_source_order() -> None:
    events: list[tuple[str, JsonObject]] = []

    async def emit(event_type: str, data: JsonObject) -> None:
        events.append((event_type, data))

    async def slow(_: JsonObject, __: ToolContext) -> ToolResult:
        await asyncio.sleep(0.02)
        return ToolResult(tool_call_id="slow", content="slow")

    async def fast(_: JsonObject, __: ToolContext) -> ToolResult:
        return ToolResult(tool_call_id="fast", content="fast")

    schema = {"type": "object", "additionalProperties": False}
    results = await execute_tool_batch(
        [
            ToolCall(id="slow", name="slow", arguments="{}"),
            ToolCall(id="fast", name="fast", arguments="{}"),
        ],
        {
            "slow": Tool("slow", "slow", schema, slow),
            "fast": Tool("fast", "fast", schema, fast),
        },
        ToolContext("run", "session", "project", "user"),
        emit,
    )

    completed_ids = [
        data["tool_call_id"] for event_type, data in events if event_type == "tool_execution_end"
    ]
    assert completed_ids == ["fast", "slow"]
    assert [result.tool_call_id for result in results] == ["slow", "fast"]


def test_context_budget_keeps_complete_recent_turns() -> None:
    messages = (
        Message(role="user", content="old" * 50),
        Message(role="assistant", content="old answer" * 50),
        Message(role="user", content="recent question"),
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall(id="call", name="read", arguments="{}"),),
        ),
        Message(role="tool", content="recent result", tool_call_id="call"),
    )
    selected, omitted = select_provider_context(messages, 100)
    assert omitted == 2
    assert selected == messages[2:]


async def test_loop_guard_stops_three_identical_tool_batches() -> None:
    class RepeatingProvider:
        async def stream(self, _: ProviderRequest) -> AsyncIterator[ProviderEvent]:
            yield ToolCallStarted(index=0, id="same", name="read")
            yield ToolCallArgumentsDelta(index=0, delta="{}")
            yield ResponseCompleted("tool_calls")

    async def execute(_: JsonObject, __: ToolContext) -> ToolResult:
        return ToolResult(tool_call_id="same", content="unchanged")

    events: list[str] = []

    async def emit(event_type: str, _: JsonObject) -> None:
        events.append(event_type)

    with pytest.raises(RuntimeError, match="repeated the same tool batch"):
        await AgentEngine(RepeatingProvider()).run(
            AgentRunRequest(
                run_id="run",
                session_id="session",
                project_id="project",
                user_id="user",
                model="model",
                system_prompt="",
                messages=(Message(role="user", content="repeat"),),
                tools=(
                    Tool(
                        name="read",
                        description="read",
                        input_schema={"type": "object", "additionalProperties": False},
                        execute=execute,
                    ),
                ),
            ),
            emit,
        )
    assert "loop_guard" in events
