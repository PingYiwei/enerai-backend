from __future__ import annotations

import json
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.modules.agents.providers.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
)
from app.modules.agents.runtime.types import (
    ImageInput,
    Message,
    ProviderRequest,
    ResponseCompleted,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallStarted,
    UsageUpdated,
)


def _data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}"


async def test_chat_completions_stream_is_normalized() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        stream = "\n\n".join(
            [
                _data({"id": "resp", "choices": [{"delta": {"content": "Hi "}}]}),
                _data(
                    {
                        "id": "resp",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "lookup",
                                                "arguments": '{"id":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ],
                    }
                ),
                _data(
                    {
                        "id": "resp",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": "1}"},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                    }
                ),
                "data: [DONE]",
            ]
        )
        return httpx.Response(
            200,
            content=(stream + "\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="secret",
        base_url="https://example.test/v1",
        http_client=http_client,
    )
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            id="openrouter",
            api_style="chat_completions",
            base_url="https://example.test/v1",
            api_key="secret",
        ),
        client,
    )
    events = [
        event
        async for event in provider.stream(
            ProviderRequest(
                model="model",
                system_prompt="System",
                messages=(
                    Message(
                        role="user",
                        content="Hello",
                        images=(
                            ImageInput(
                                "att",
                                "image.png",
                                "image/png",
                                "https://minio.enerai.cloud/enerai/image?signature=test",
                            ),
                        ),
                    ),
                ),
            )
        )
    ]
    await client.close()

    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "System"}
    assert captured["body"]["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://minio.enerai.cloud/enerai/image?signature=test"
        },
    }
    assert TextDelta("Hi ") in events
    assert ToolCallStarted(index=0, id="call_1", name="lookup") in events
    assert [event.delta for event in events if isinstance(event, ToolCallArgumentsDelta)] == [
        '{"id":',
        "1}",
    ]
    assert any(isinstance(event, UsageUpdated) for event in events)
    assert events[-1] == ResponseCompleted("tool_calls", "resp")


async def test_responses_stream_is_normalized() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        stream = "\n\n".join(
            [
                _data({"type": "response.output_text.delta", "delta": "Done"}),
                _data(
                    {
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {
                            "id": "item_1",
                            "type": "function_call",
                            "call_id": "call_2",
                            "name": "lookup",
                            "arguments": "",
                        },
                    }
                ),
                _data(
                    {
                        "type": "response.function_call_arguments.delta",
                        "sequence_number": 2,
                        "output_index": 0,
                        "item_id": "item_1",
                        "delta": '{"id":1}',
                    }
                ),
                _data(
                    {
                        "type": "response.completed",
                        "sequence_number": 3,
                        "response": {
                            "id": "resp_2",
                            "created_at": 1,
                            "model": "model",
                            "object": "response",
                            "output": [],
                            "parallel_tool_calls": True,
                            "status": "completed",
                            "tool_choice": "auto",
                            "tools": [],
                            "usage": {
                                "input_tokens": 8,
                                "input_tokens_details": {
                                    "cached_tokens": 0,
                                    "cache_write_tokens": 0,
                                },
                                "output_tokens": 2,
                                "output_tokens_details": {"reasoning_tokens": 0},
                                "total_tokens": 10,
                            },
                        },
                    }
                ),
            ]
        )
        return httpx.Response(
            200,
            content=(stream + "\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="secret",
        base_url="https://api.openai.test/v1",
        http_client=http_client,
    )
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            id="openai",
            api_style="responses",
            base_url="https://api.openai.test/v1",
            api_key="secret",
        ),
        client,
    )
    events = [
        event
        async for event in provider.stream(
            ProviderRequest(
                model="model",
                system_prompt="System",
                messages=(
                    Message(
                        role="user",
                        content="Hello",
                        images=(
                            ImageInput(
                                "att",
                                "image.png",
                                "image/png",
                                "https://minio.enerai.cloud/enerai/image?signature=test",
                            ),
                        ),
                    ),
                ),
            )
        )
    ]
    await client.close()

    assert captured["path"] == "/v1/responses"
    assert captured["body"]["instructions"] == "System"
    assert captured["body"]["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": "https://minio.enerai.cloud/enerai/image?signature=test",
    }
    assert events[0] == TextDelta("Done")
    assert events[1] == ToolCallStarted(index=0, id="call_2", name="lookup")
    assert events[2] == ToolCallArgumentsDelta(index=0, delta='{"id":1}')
    assert isinstance(events[3], UsageUpdated)
    assert events[-1] == ResponseCompleted("stop", "resp_2")
