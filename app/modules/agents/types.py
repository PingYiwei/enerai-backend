from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

type JsonObject = dict[str, Any]
type MessageRole = Literal["user", "assistant", "tool"]
type StopReason = Literal["stop", "length", "tool_calls", "cancelled", "error"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ImageInput:
    id: str
    name: str
    media_type: str
    data_base64: str | None = None


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    images: tuple[ImageInput, ...] = ()


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass(frozen=True, slots=True)
class ProviderTool:
    name: str
    description: str
    input_schema: JsonObject


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    model: str
    messages: tuple[Message, ...]
    system_prompt: str = ""
    tools: tuple[ProviderTool, ...] = ()
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class TextDelta:
    delta: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    delta: str


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    index: int
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class ToolCallArgumentsDelta:
    index: int
    delta: str


@dataclass(frozen=True, slots=True)
class UsageUpdated:
    usage: Usage


@dataclass(frozen=True, slots=True)
class ResponseCompleted:
    stop_reason: StopReason
    response_id: str | None = None


type ProviderEvent = (
    TextDelta
    | ReasoningDelta
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | UsageUpdated
    | ResponseCompleted
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    details: JsonObject = field(default_factory=dict)
    terminate: bool = False


@dataclass(frozen=True, slots=True)
class AgentResult:
    messages: tuple[Message, ...]
    final_message: Message
    usage: Usage
    turns: int
