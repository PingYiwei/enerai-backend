from fastapi import APIRouter

from app.modules.agents.router import router as agents_router
from app.modules.auth.router import router as auth_router
from app.modules.inspections.router import router as inspections_router
from app.modules.optimizer.router import router as optimizer_router
from app.modules.projects.router import router as projects_router
from app.modules.studio.router import router as studio_router

api_router = APIRouter()
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(projects_router, prefix="/projects", tags=["projects"])
api_router.include_router(
    studio_router,
    prefix="/projects/{project_id}/studio",
    tags=["studio"],
)
api_router.include_router(
    optimizer_router,
    prefix="/projects/{project_id}/optimizer",
    tags=["optimizer"],
)
api_router.include_router(
    inspections_router,
    prefix="/projects/{project_id}/inspections",
    tags=["inspections"],
)
api_router.include_router(agents_router, tags=["agents"])
