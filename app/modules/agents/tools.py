from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from app.modules.agents.types import JsonObject, ProviderTool, ToolCall, ToolResult

type ToolEffect = Literal["read", "write", "external", "compute"]
type ExecutionMode = Literal["parallel", "sequential"]
type ResultVisibility = Literal["model", "ui", "both", "reference"]
type ToolExecutor = Callable[[JsonObject, ToolContext], Awaitable[ToolResult]]
type ToolEventSink = Callable[[str, JsonObject], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ToolContext:
    run_id: str
    session_id: str
    project_id: str
    user_id: str


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: JsonObject
    execute: ToolExecutor
    effect: ToolEffect = "read"
    execution_mode: ExecutionMode = "parallel"
    result_visibility: ResultVisibility = "both"
    idempotent: bool = True

    def provider_spec(self) -> ProviderTool:
        return ProviderTool(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )


def _arguments(call: ToolCall, tool: Tool) -> JsonObject:
    try:
        value = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as error:
        raise ValidationError(f"Tool arguments are not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValidationError("Tool arguments must be a JSON object")
    try:
        Draft202012Validator.check_schema(tool.input_schema)
    except SchemaError as error:
        raise RuntimeError(f"Tool {tool.name} has an invalid input schema") from error
    Draft202012Validator(tool.input_schema).validate(value)
    return value


async def execute_tool_batch(
    calls: Sequence[ToolCall],
    tools: Mapping[str, Tool],
    context: ToolContext,
    emit: ToolEventSink,
) -> list[ToolResult]:
    prepared: list[tuple[int, ToolCall, Tool, JsonObject] | ToolResult] = []
    requires_sequential = False

    for index, call in enumerate(calls):
        tool = tools.get(call.name)
        if tool is None:
            prepared.append(
                ToolResult(
                    tool_call_id=call.id,
                    content=f"Unknown tool: {call.name}",
                    is_error=True,
                )
            )
            continue
        try:
            arguments = _arguments(call, tool)
        except ValidationError as error:
            prepared.append(
                ToolResult(
                    tool_call_id=call.id,
                    content=f"Invalid arguments for {call.name}: {error.message}",
                    is_error=True,
                )
            )
            continue
        requires_sequential = requires_sequential or tool.execution_mode == "sequential"
        prepared.append((index, call, tool, arguments))

    async def run_one(
        index: int,
        call: ToolCall,
        tool: Tool,
        arguments: JsonObject,
    ) -> tuple[int, ToolResult]:
        await emit(
            "tool_execution_start",
            {
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": arguments,
                "effect": tool.effect,
            },
        )
        try:
            result = await tool.execute(arguments, context)
            if result.tool_call_id != call.id:
                result = replace(result, tool_call_id=call.id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            result = ToolResult(
                tool_call_id=call.id,
                content=f"{type(error).__name__}: {error}",
                is_error=True,
            )
        await emit(
            "tool_execution_end",
            {
                "tool_call_id": call.id,
                "name": call.name,
                "content": (result.content if tool.result_visibility in {"ui", "both"} else ""),
                "is_error": result.is_error,
                "details": (result.details if tool.result_visibility in {"ui", "both"} else {}),
            },
        )
        return index, result

    completed: dict[int, ToolResult] = {}
    for index, item in enumerate(prepared):
        if isinstance(item, ToolResult):
            completed[index] = item
            await emit(
                "tool_execution_end",
                {
                    "tool_call_id": item.tool_call_id,
                    "content": item.content,
                    "is_error": True,
                    "details": {},
                },
            )

    runnable = [item for item in prepared if not isinstance(item, ToolResult)]
    if requires_sequential:
        for item in runnable:
            index, result = await run_one(*item)
            completed[index] = result
    else:
        tasks = [asyncio.create_task(run_one(*item)) for item in runnable]
        for task in asyncio.as_completed(tasks):
            index, result = await task
            completed[index] = result

    return [completed[index] for index in range(len(calls))]
