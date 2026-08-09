from fastapi import APIRouter

from app.api.dependencies import CurrentPrincipal, Database
from app.modules.studio.schemas import StudioCatalog, StudioGraph, StudioGraphUpdate
from app.modules.studio.service import CATALOG, get_graph, save_graph

router = APIRouter()


@router.get("/catalog", response_model=StudioCatalog)
async def catalog() -> StudioCatalog:
    return CATALOG


@router.get("/graph", response_model=StudioGraph)
async def graph(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> StudioGraph:
    return await get_graph(database, principal, project_id)


@router.put("/graph", response_model=StudioGraph)
async def update_graph(
    project_id: str,
    body: StudioGraphUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> StudioGraph:
    return await save_graph(database, principal, project_id, body)
