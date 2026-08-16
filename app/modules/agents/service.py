from __future__ import annotations

import asyncio
from contextlib import suppress

from app.core.config import Settings
from app.core.errors import AppError
from app.core.security import Principal
from app.modules.agents.prompts import render_agent_system_prompt
from app.modules.agents.providers.openai_compatible import OpenAICompatibleProvider
from app.modules.agents.providers.registry import ProviderId, ProviderRegistry
from app.modules.agents.runtime.engine import AgentEngine, AgentRunRequest
from app.modules.agents.runtime.titles import generate_session_title
from app.modules.agents.runtime.types import JsonObject, Message, TraceContext
from app.modules.agents.schemas import RunAccepted, RunCreate
from app.modules.agents.storage.artifacts import artifact_tools
from app.modules.agents.storage.repository import MongoAgentRepository
from app.modules.agents.tools.base import Tool
from app.modules.agents.tools.project import project_tools
from app.modules.agents.tools.studio import studio_tools
from app.modules.auth.model_settings import configured_auxiliary_model, resolve_provider_runtime


class AgentRunCoordinator:
    def __init__(self, settings: Settings, providers: ProviderRegistry) -> None:
        self._settings = settings
        self._providers = providers
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._title_tasks: set[asyncio.Task[None]] = set()

    async def start(
        self,
        repository: MongoAgentRepository,
        principal: Principal,
        session_id: str,
        request: RunCreate,
    ) -> RunAccepted:
        runtime = await resolve_provider_runtime(
            repository.database,
            principal.user_id,
            self._settings,
            requested_provider=request.provider,
            requested_api_style=request.api_style,
            requested_model=request.model,
            multimodal=bool(request.attachment_ids),
        )
        provider_id: ProviderId = runtime.provider
        model = runtime.model
        if not model:
            raise AppError(
                "model_not_configured",
                "A model must be selected for this run",
                status_code=422,
            )
        provider = self._providers.get(
            provider_id,
            runtime.api_style,
            api_key=runtime.api_key,
            base_url=runtime.base_url,
        )
        title_target = await repository.automatic_title_target(principal, session_id)
        auxiliary_model = (
            await configured_auxiliary_model(
                repository.database, principal.user_id, provider_id
            )
            if title_target is not None
            else ""
        )
        title_mode = (
            "auxiliary"
            if title_target is not None and auxiliary_model
            else "primary_fallback"
            if title_target is not None
            else "skipped"
        )
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
        if title_target is not None:
            title_task = asyncio.create_task(
                self._generate_title(
                    repository=repository,
                    run_id=accepted.id,
                    session_id=session_id,
                    owner_id=principal.user_id,
                    expected_title=title_target,
                    user_message=request.content,
                    provider=provider,
                    model=auxiliary_model or model,
                ),
                name=f"session-title:{accepted.id}",
            )
            self._track_title(title_task)
        return accepted.model_copy(update={"title_generation": title_mode})

    async def recover(self, repository: MongoAgentRepository) -> tuple[int, int]:
        accepted, interrupted = await repository.recover_incomplete_runs()
        recovered = 0
        unavailable = 0
        for operation in accepted:
            run_id = str(operation["_id"])
            try:
                runtime = await resolve_provider_runtime(
                    repository.database,
                    str(operation["owner_id"]),
                    self._settings,
                    requested_provider=operation["provider"],
                    requested_api_style=operation.get("api_style"),
                    requested_model=str(operation["model"]),
                    multimodal=False,
                )
            except AppError as error:
                unavailable += 1
                await repository.finish_run(
                    run_id,
                    "failed",
                    error=f"{error.code}: {error.message}",
                )
                continue
            provider = self._providers.get(
                runtime.provider,
                runtime.api_style,
                api_key=runtime.api_key,
                base_url=runtime.base_url,
            )
            task = asyncio.create_task(
                self._execute(repository, run_id, provider, str(operation["model"])),
                name=f"agent-run:{run_id}",
            )
            self._track(run_id, task)
            recovered += 1
        return recovered, len(interrupted) + unavailable

    def _track(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks[run_id] = task

        def forget(completed: asyncio.Task[None]) -> None:
            if self._tasks.get(run_id) is completed:
                self._tasks.pop(run_id, None)

        task.add_done_callback(forget)

    def _track_title(self, task: asyncio.Task[None]) -> None:
        self._title_tasks.add(task)
        task.add_done_callback(self._title_tasks.discard)

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
        tasks = [*self._tasks.values(), *self._title_tasks]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._title_tasks.clear()

    async def _generate_title(
        self,
        *,
        repository: MongoAgentRepository,
        run_id: str,
        session_id: str,
        owner_id: str,
        expected_title: str,
        user_message: str,
        provider: OpenAICompatibleProvider,
        model: str,
    ) -> None:
        try:
            title = await generate_session_title(
                provider,
                model,
                user_message,
                TraceContext(
                    user_id=owner_id,
                    source="session_title",
                    feature="automatic_session_title",
                    session_id=session_id,
                    run_id=run_id,
                    tags=("auxiliary",),
                ),
            )
            if not title:
                raise ValueError("Title model returned an empty title")
            updated = await repository.apply_generated_title(
                session_id, owner_id, expected_title, title
            )
            if updated:
                await repository.append_event(
                    run_id,
                    "session_title_updated",
                    {"session_id": session_id, "title": title},
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with suppress(Exception):
                await repository.append_event(
                    run_id,
                    "session_title_failed",
                    {"session_id": session_id, "error": type(error).__name__},
                )

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
                    system_prompt=render_agent_system_prompt(
                        str(operation.get("surface", "insight")),
                        timezone_name=self._settings.agent_timezone,
                    ),
                    messages=tuple(messages),
                    tools=self._tools(repository, str(operation.get("surface", "insight"))),
                    context_char_budget=self._settings.agent_context_char_budget,
                    source=str(operation.get("surface", "insight")),
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
