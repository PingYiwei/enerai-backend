from __future__ import annotations

import asyncio

from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.artifacts import artifact_tools
from app.modules.agents.engine import AgentEngine, AgentRunRequest
from app.modules.agents.project_tools import project_tools
from app.modules.agents.providers.openai_compatible import OpenAICompatibleProvider
from app.modules.agents.providers.registry import ProviderId, ProviderRegistry
from app.modules.agents.repository import MongoAgentRepository
from app.modules.agents.schemas import RunAccepted, RunCreate
from app.modules.agents.studio_tools import studio_tools
from app.modules.agents.tools import Tool
from app.modules.agents.types import JsonObject, Message


class AgentRunCoordinator:
    def __init__(self, settings: Settings, providers: ProviderRegistry) -> None:
        self._settings = settings
        self._providers = providers
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(
        self,
        repository: MongoAgentRepository,
        principal: Principal,
        session_id: str,
        request: RunCreate,
    ) -> RunAccepted:
        provider_id: ProviderId = request.provider or self._settings.default_provider
        model = (request.model or self._settings.default_model).strip()
        if not model:
            raise AppError(
                "model_not_configured",
                "A model must be selected for this run",
                status_code=422,
            )
        provider = self._providers.get(provider_id, request.api_style)
        accepted = await repository.accept_run(
            principal,
            session_id,
            request,
            provider_id,
            model,
        )
        task = asyncio.create_task(
            self._execute(repository, accepted.id, provider, model),
            name=f"agent-run:{accepted.id}",
        )
        self._track(accepted.id, task)
        return accepted

    async def recover(self, repository: MongoAgentRepository) -> tuple[int, int]:
        accepted, interrupted = await repository.recover_incomplete_runs()
        for operation in accepted:
            run_id = str(operation["_id"])
            provider = self._providers.get(
                operation["provider"],
                operation.get("api_style"),
            )
            task = asyncio.create_task(
                self._execute(repository, run_id, provider, str(operation["model"])),
                name=f"agent-run:{run_id}",
            )
            self._track(run_id, task)
        return len(accepted), len(interrupted)

    def _track(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[run_id] = task

        def forget(completed: asyncio.Task[None]) -> None:
            if self._tasks.get(run_id) is completed:
                self._tasks.pop(run_id, None)

        task.add_done_callback(forget)

    async def abort(
        self,
        repository: MongoAgentRepository,
        principal: Principal,
        run_id: str,
    ) -> None:
        run = await repository.get_run(principal, run_id)
        if run.status in {"completed", "failed", "cancelled"}:
            return
        await repository.append_event(run_id, "abort_requested", {})
        task = self._tasks.get(run_id)
        if task is not None:
            task.cancel()
            return
        await repository.finish_run(run_id, "cancelled")

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _execute(
        self,
        repository: MongoAgentRepository,
        run_id: str,
        provider: OpenAICompatibleProvider,
        model: str,
    ) -> None:
        try:
            operation, messages = await repository.run_context(run_id)
            initial_count = len(messages)
            if not await repository.mark_running(run_id):
                return

            async def emit(event_type: str, data: JsonObject) -> None:
                await repository.append_event(run_id, event_type, data)

            async def checkpoint(current_messages: tuple[Message, ...], _: int) -> None:
                await repository.persist_checkpoint(
                    run_id,
                    current_messages,
                    initial_count,
                )

            result = await AgentEngine(provider).run(
                AgentRunRequest(
                    run_id=run_id,
                    session_id=operation["session_id"],
                    project_id=operation["project_id"],
                    user_id=operation["owner_id"],
                    model=model,
                    system_prompt=self._system_prompt(str(operation.get("surface", "insight"))),
                    messages=tuple(messages),
                    tools=self._tools(repository, str(operation.get("surface", "insight"))),
                    context_char_budget=self._settings.agent_context_char_budget,
                ),
                emit,
                checkpoint,
            )
            await repository.finish_run(run_id, "completed", result.usage)
        except asyncio.CancelledError:
            await repository.finish_run(run_id, "cancelled")
            raise
        except Exception as error:
            await repository.finish_run(
                run_id,
                "failed",
                error=f"{type(error).__name__}: {error}",
            )

    def _tools(self, repository: MongoAgentRepository, surface: str) -> tuple[Tool, ...]:
        shared = project_tools(repository.database) + artifact_tools(repository.database)
        return shared + studio_tools(repository.database) if surface == "studio" else shared

    @staticmethod
    def _system_prompt(surface: str) -> str:
        if surface == "studio":
            return (
                "You are Nodex Studio Agent. Build valid energy-equipment graphs using complete "
                "replace_studio_graph operations. Preserve node positions unless the user asks for "
                "layout changes. Explain modeling decisions briefly and never invent live data."
            )
        return (
            "You are Nodex Insight, an energy-system analysis agent. Be precise, distinguish "
            "evidence from inference, and use project tools when data is required."
        )
