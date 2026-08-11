from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import httpx

from app.modules.agents.runtime.types import (
    ImageInput,
    Message,
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

ApiStyle = Literal["responses", "chat_completions"]


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    id: Literal["openai", "openrouter", "bailian"]
    api_style: ApiStyle
    base_url: str
    api_key: str
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: OpenAICompatibleConfig,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        if self._config.api_style == "responses":
            async for event in self._stream_responses(request):
                yield event
            return
        async for event in self._stream_chat_completions(request):
            yield event

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            **self._config.headers,
        }

    async def _sse(self, path: str, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        url = f"{self._config.base_url.rstrip('/')}/{path.lstrip('/')}"
        async with self._client.stream(
            "POST",
            url,
            headers=self._headers(),
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                decoded = json.loads(data)
                if isinstance(decoded, dict):
                    yield decoded

    async def _stream_chat_completions(
        self, request: ProviderRequest
    ) -> AsyncIterator[ProviderEvent]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": _chat_messages(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        started_calls: set[int] = set()
        response_id: str | None = None
        finish_reason = "stop"
        async for chunk in self._sse("chat/completions", payload):
            response_id = cast(str | None, chunk.get("id")) or response_id
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                yield UsageUpdated(_chat_usage(usage))
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            finish_reason = _stop_reason(choice.get("finish_reason") or finish_reason)
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                yield TextDelta(content)
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                yield ReasoningDelta(reasoning)
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for raw_call in tool_calls:
                if not isinstance(raw_call, dict):
                    continue
                index = int(raw_call.get("index", 0))
                function = raw_call.get("function")
                function = function if isinstance(function, dict) else {}
                call_id = str(raw_call.get("id") or "")
                name = str(function.get("name") or "")
                if index not in started_calls:
                    started_calls.add(index)
                    yield ToolCallStarted(index=index, id=call_id, name=name)
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    yield ToolCallArgumentsDelta(index=index, delta=arguments)
        yield ResponseCompleted(stop_reason=_stop_reason(finish_reason), response_id=response_id)

    async def _stream_responses(self, request: ProviderRequest) -> AsyncIterator[ProviderEvent]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": _responses_input(request.messages),
            "stream": True,
        }
        if request.system_prompt:
            payload["instructions"] = request.system_prompt
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": True,
                }
                for tool in request.tools
            ]
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens

        item_indexes: dict[str, int] = {}
        next_index = 0
        async for event in self._sse("responses", payload):
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield TextDelta(delta)
            elif event_type in {
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            }:
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield ReasoningDelta(delta)
            elif event_type == "response.output_item.added":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    item_id = str(item.get("id") or item.get("call_id") or next_index)
                    index = int(event.get("output_index", next_index))
                    next_index = max(next_index, index + 1)
                    item_indexes[item_id] = index
                    yield ToolCallStarted(
                        index=index,
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                    )
            elif event_type == "response.function_call_arguments.delta":
                item_id = str(event.get("item_id") or "")
                index = item_indexes.get(item_id, int(event.get("output_index", 0)))
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield ToolCallArgumentsDelta(index=index, delta=delta)
            elif event_type == "response.completed":
                response = event.get("response")
                if isinstance(response, dict):
                    usage = response.get("usage")
                    if isinstance(usage, dict):
                        yield UsageUpdated(_responses_usage(usage))
                    yield ResponseCompleted(
                        stop_reason=_responses_stop_reason(response),
                        response_id=str(response.get("id") or "") or None,
                    )
            elif event_type in {"response.failed", "error"}:
                error = event.get("error") or event.get("response") or event
                raise RuntimeError(f"Provider response failed: {error}")


def _chat_messages(request: ProviderRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for message in request.messages:
        if message.role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
            continue
        content: str | list[dict[str, Any]] = message.content
        if message.images:
            content = [{"type": "text", "text": message.content}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": _required_image_url(image)},
                }
                for image in message.images
            )
        encoded: dict[str, Any] = {"role": message.role, "content": content}
        if message.tool_calls:
            encoded["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        messages.append(encoded)
    return messages


def _responses_input(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue
        if message.content or message.images:
            if message.images:
                image_content: list[dict[str, Any]] = [
                    {"type": "input_text", "text": message.content}
                ]
                image_content.extend(
                    {
                        "type": "input_image",
                        "image_url": _required_image_url(image),
                    }
                    for image in message.images
                )
                content: str | list[dict[str, Any]] = image_content
            else:
                content = message.content
            items.append({"role": message.role, "content": content})
        items.extend(
            {
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        )
    return items


def _required_image_url(image: ImageInput) -> str:
    if image.url is None:
        raise ValueError(f"Image {image.id} does not have a provider-accessible URL")
    return image.url


def _chat_usage(raw: dict[str, Any]) -> Usage:
    details = raw.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    completion = raw.get("completion_tokens_details")
    completion = completion if isinstance(completion, dict) else {}
    return Usage(
        input_tokens=int(raw.get("prompt_tokens", 0)),
        output_tokens=int(raw.get("completion_tokens", 0)),
        cached_input_tokens=int(details.get("cached_tokens", 0)),
        reasoning_tokens=int(completion.get("reasoning_tokens", 0)),
    )


def _responses_usage(raw: dict[str, Any]) -> Usage:
    input_details = raw.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = raw.get("output_tokens_details")
    output_details = output_details if isinstance(output_details, dict) else {}
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cached_input_tokens=int(input_details.get("cached_tokens", 0)),
        reasoning_tokens=int(output_details.get("reasoning_tokens", 0)),
    )


def _stop_reason(value: Any) -> Literal["stop", "length", "tool_calls", "error"]:
    if value in {"length", "max_tokens", "incomplete"}:
        return "length"
    if value in {"tool_calls", "function_call"}:
        return "tool_calls"
    if value in {"error", "failed"}:
        return "error"
    return "stop"


def _responses_stop_reason(response: dict[str, Any]) -> Literal["stop", "length", "error"]:
    if response.get("status") == "failed":
        return "error"
    if response.get("status") == "incomplete":
        return "length"
    return "stop"
