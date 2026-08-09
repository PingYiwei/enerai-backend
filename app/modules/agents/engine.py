from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.modules.agents.provider import Provider
from app.modules.agents.tools import Tool, ToolContext, execute_tool_batch
from app.modules.agents.types import (
    AgentResult,
    JsonObject,
    Message,
    ProviderRequest,
    ReasoningDelta,
    ResponseCompleted,
    TextDelta,
    ToolCall,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    Usage,
    UsageUpdated,
)

type AgentEventSink = Callable[[str, JsonObject], Awaitable[None]]
type AgentCheckpointSink = Callable[[tuple[Message, ...], int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    run_id: str
    session_id: str
    project_id: str
    user_id: str
    model: str
    system_prompt: str
    messages: tuple[Message, ...]
    tools: tuple[Tool, ...] = ()
    max_turns: int = 32
    max_output_tokens: int | None = None
    temperature: float | None = None
    context_char_budget: int = 300_000


@dataclass(slots=True)
class _ToolCallBuffer:
    id: str = ""
    name: str = ""
    arguments: str = ""


class AgentEngine:
    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    async def run(
        self,
        request: AgentRunRequest,
        emit: AgentEventSink,
        checkpoint: AgentCheckpointSink | None = None,
    ) -> AgentResult:
        messages = list(request.messages)
        usage = Usage()
        tools = {tool.name: tool for tool in request.tools}
        recent_tool_batches: list[tuple[tuple[str, str], ...]] = []
        await emit("run_start", {"run_id": request.run_id})

        try:
            for turn in range(1, request.max_turns + 1):
                await emit("turn_start", {"turn": turn})
                text_parts: list[str] = []
                tool_buffers: dict[int, _ToolCallBuffer] = {}
                stop_reason = "stop"

                provider_messages, omitted = select_provider_context(
                    tuple(messages), request.context_char_budget
                )
                if omitted:
                    await emit(
                        "context_pruned",
                        {"omitted_messages": omitted, "turn": turn},
                    )
                provider_request = ProviderRequest(
                    model=request.model,
                    messages=provider_messages,
                    system_prompt=request.system_prompt,
                    tools=tuple(tool.provider_spec() for tool in request.tools),
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
                await emit("message_start", {"role": "assistant", "turn": turn})

                async for event in self._provider.stream(provider_request):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.delta)
                        await emit("message_delta", {"delta": event.delta, "turn": turn})
                    elif isinstance(event, ReasoningDelta):
                        await emit("reasoning_delta", {"delta": event.delta, "turn": turn})
                    elif isinstance(event, ToolCallStarted):
                        buffer = tool_buffers.setdefault(event.index, _ToolCallBuffer())
                        buffer.id = event.id or buffer.id
                        buffer.name = event.name or buffer.name
                    elif isinstance(event, ToolCallArgumentsDelta):
                        tool_buffers.setdefault(
                            event.index, _ToolCallBuffer()
                        ).arguments += event.delta
                    elif isinstance(event, UsageUpdated):
                        usage = usage + event.usage
                        await emit(
                            "usage_update",
                            {
                                "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens,
                                "cached_input_tokens": usage.cached_input_tokens,
                                "reasoning_tokens": usage.reasoning_tokens,
                            },
                        )
                    elif isinstance(event, ResponseCompleted):
                        stop_reason = event.stop_reason

                tool_calls = tuple(
                    ToolCall(id=buffer.id, name=buffer.name, arguments=buffer.arguments)
                    for _, buffer in sorted(tool_buffers.items())
                )
                if tool_calls:
                    stop_reason = "tool_calls"
                assistant = Message(
                    role="assistant",
                    content="".join(text_parts),
                    tool_calls=tool_calls,
                )
                messages.append(assistant)
                await emit(
                    "message_end",
                    {
                        "role": "assistant",
                        "content": assistant.content,
                        "tool_calls": [
                            {"id": call.id, "name": call.name, "arguments": call.arguments}
                            for call in tool_calls
                        ],
                        "stop_reason": stop_reason,
                        "turn": turn,
                    },
                )
                if checkpoint is not None:
                    await checkpoint(tuple(messages), turn)

                if not tool_calls:
                    await emit("turn_end", {"turn": turn, "tool_count": 0})
                    await emit("run_end", {"run_id": request.run_id, "outcome": "completed"})
                    return AgentResult(
                        messages=tuple(messages),
                        final_message=assistant,
                        usage=usage,
                        turns=turn,
                    )

                fingerprint = tuple((call.name, call.arguments) for call in tool_calls)
                recent_tool_batches.append(fingerprint)
                if len(recent_tool_batches) >= 3 and len(set(recent_tool_batches[-3:])) == 1:
                    await emit(
                        "loop_guard",
                        {
                            "turn": turn,
                            "reason": "repeated_tool_batch",
                            "tool_names": [call.name for call in tool_calls],
                        },
                    )
                    raise RuntimeError("Agent repeated the same tool batch three times")

                results = await execute_tool_batch(
                    tool_calls,
                    tools,
                    ToolContext(
                        run_id=request.run_id,
                        session_id=request.session_id,
                        project_id=request.project_id,
                        user_id=request.user_id,
                    ),
                    emit,
                )
                for result in results:
                    messages.append(
                        Message(
                            role="tool",
                            content=result.content,
                            tool_call_id=result.tool_call_id,
                        )
                    )
                if checkpoint is not None:
                    await checkpoint(tuple(messages), turn)
                await emit("turn_end", {"turn": turn, "tool_count": len(results)})

                if results and all(result.terminate for result in results):
                    await emit("run_end", {"run_id": request.run_id, "outcome": "completed"})
                    return AgentResult(
                        messages=tuple(messages),
                        final_message=assistant,
                        usage=usage,
                        turns=turn,
                    )

            raise RuntimeError(f"Agent exceeded max_turns={request.max_turns}")
        except asyncio.CancelledError:
            await emit("run_end", {"run_id": request.run_id, "outcome": "cancelled"})
            raise
        except Exception as error:
            await emit(
                "run_end",
                {
                    "run_id": request.run_id,
                    "outcome": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


def select_provider_context(
    messages: tuple[Message, ...], char_budget: int
) -> tuple[tuple[Message, ...], int]:
    if sum(_message_size(message) for message in messages) <= char_budget:
        return messages, 0
    groups: list[list[Message]] = []
    for message in messages:
        if message.role == "user" or not groups:
            groups.append([])
        groups[-1].append(message)
    selected: list[list[Message]] = []
    used = 0
    for group in reversed(groups):
        size = sum(_message_size(message) for message in group)
        if selected and used + size > char_budget:
            break
        selected.append(group)
        used += size
    selected.reverse()
    projected = tuple(message for group in selected for message in group)
    return projected, len(messages) - len(projected)


def _message_size(message: Message) -> int:
    return (
        len(message.content)
        + sum(len(call.id) + len(call.name) + len(call.arguments) for call in message.tool_calls)
        + sum(len(image.data_base64 or "") for image in message.images)
    )
