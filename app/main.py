from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.database import database_lifespan
from app.core.errors import install_error_handlers
from app.modules.agents.providers.registry import ProviderRegistry
from app.modules.agents.service import AgentRunCoordinator
from app.modules.agents.storage.attachments import cleanup_expired_drafts
from app.modules.agents.storage.repository import MongoAgentRepository
from app.modules.inspections.scheduler import InspectionScheduler


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        providers = ProviderRegistry()
        coordinator = AgentRunCoordinator(resolved, providers)
        async with database_lifespan(resolved) as database:
            inspection_scheduler = InspectionScheduler(database)
            app.state.database = database
            app.state.agent_coordinator = coordinator
            await cleanup_expired_drafts(database)
            await coordinator.recover(MongoAgentRepository(database))
            inspection_scheduler.start()
            try:
                yield
            finally:
                await inspection_scheduler.close()
                await coordinator.close()
                await providers.close()

    app = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    app.include_router(api_router, prefix=resolved.api_prefix)

    @app.get("/health/live", tags=["health"])
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def ready(request: Request) -> dict[str, str]:
        await request.app.state.database.command("ping")
        return {"status": "ready"}

    return app


app = create_app()
