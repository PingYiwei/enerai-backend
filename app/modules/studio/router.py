from fastapi import APIRouter

from app.api.dependencies import CurrentPrincipal, Database
from app.modules.studio.schemas import (
    EngineeringParameterCatalog,
    StudioCatalog,
    StudioCategories,
    StudioGraph,
    StudioGraphUpdate,
)
from app.modules.studio.service import (
    CATALOG,
    get_categories,
    get_engineering_parameter_catalog,
    get_graph,
    save_graph,
)

router = APIRouter()


@router.get("/catalog", response_model=StudioCatalog)
async def catalog() -> StudioCatalog:
    return CATALOG


@router.get("/categories", response_model=StudioCategories)
async def categories(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> StudioCategories:
    return await get_categories(database, principal, project_id)


@router.get("/engineering-parameters", response_model=EngineeringParameterCatalog)
async def engineering_parameters(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> EngineeringParameterCatalog:
    return await get_engineering_parameter_catalog(database, principal, project_id)


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
