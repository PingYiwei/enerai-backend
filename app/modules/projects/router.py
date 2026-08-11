from urllib.parse import quote

from fastapi import APIRouter, Query, Response, status

from app.api.dependencies import CurrentPrincipal, Database, Projects
from app.modules.projects.data import (
    cleanup_project_resources,
    get_data_source,
    owned_project,
    point_scheme_xlsx,
    project_point_scheme,
    project_rdf,
    properties,
    query_data,
    save_data_source,
    test_data_source,
)
from app.modules.projects.schemas import (
    DataQuery,
    DataQueryResult,
    DataSourceTestResult,
    DataSourceUpdate,
    DataSourceView,
    PointScheme,
    ProjectCreate,
    ProjectDetail,
    ProjectList,
    ProjectTokenUsage,
    ProjectUpdate,
    PropertyCatalog,
)
from app.modules.projects.service import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    project_token_usage,
    update_project,
)

router = APIRouter()


@router.post("", response_model=ProjectDetail, status_code=status.HTTP_201_CREATED)
async def create(
    request: ProjectCreate,
    projects: Projects,
    principal: CurrentPrincipal,
) -> ProjectDetail:
    return await create_project(projects, principal, request)


@router.get("", response_model=ProjectList)
async def list_all(projects: Projects, principal: CurrentPrincipal) -> ProjectList:
    return await list_projects(projects, principal)


@router.get("/{project_id}", response_model=ProjectDetail)
async def get_one(
    project_id: str,
    projects: Projects,
    principal: CurrentPrincipal,
) -> ProjectDetail:
    return await get_project(projects, principal, project_id)


@router.get("/{project_id}/token-usage", response_model=ProjectTokenUsage)
async def token_usage(
    project_id: str,
    projects: Projects,
    database: Database,
    principal: CurrentPrincipal,
    days: int = Query(default=14, ge=1, le=90),
    timezone_offset_minutes: int = Query(default=0, ge=-840, le=840),
) -> ProjectTokenUsage:
    return await project_token_usage(
        projects,
        database,
        principal,
        project_id,
        days=days,
        timezone_offset_minutes=timezone_offset_minutes,
    )


@router.patch("/{project_id}", response_model=ProjectDetail)
async def update(
    project_id: str,
    request: ProjectUpdate,
    projects: Projects,
    principal: CurrentPrincipal,
) -> ProjectDetail:
    return await update_project(projects, principal, project_id, request)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(
    project_id: str,
    projects: Projects,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    await delete_project(projects, principal, project_id)
    await cleanup_project_resources(database, principal.user_id, project_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{project_id}/data-source", response_model=DataSourceView)
async def data_source(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> DataSourceView:
    return await get_data_source(database, principal, project_id)


@router.put("/{project_id}/data-source", response_model=DataSourceView)
async def update_data_source(
    project_id: str,
    body: DataSourceUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> DataSourceView:
    return await save_data_source(database, principal, project_id, body)


@router.post("/{project_id}/data-source/test", response_model=DataSourceTestResult)
async def test_source(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> DataSourceTestResult:
    return await test_data_source(database, principal, project_id)


@router.get("/{project_id}/properties", response_model=PropertyCatalog)
async def property_catalog(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> PropertyCatalog:
    return await properties(database, principal, project_id)


@router.post("/{project_id}/data/query", response_model=DataQueryResult)
async def data_query(
    project_id: str,
    body: DataQuery,
    database: Database,
    principal: CurrentPrincipal,
) -> DataQueryResult:
    return await query_data(database, principal, project_id, body)


@router.get("/{project_id}/schema/rdf")
async def rdf(project_id: str, database: Database, principal: CurrentPrincipal) -> Response:
    project = await owned_project(database, principal, project_id)
    return Response(project_rdf(project), media_type="text/turtle; charset=utf-8")


@router.get("/{project_id}/point-scheme", response_model=PointScheme)
async def points(project_id: str, database: Database, principal: CurrentPrincipal) -> PointScheme:
    project = await owned_project(database, principal, project_id)
    return await project_point_scheme(database, project)


@router.get("/{project_id}/point-scheme/export")
async def export_points(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> Response:
    project = await owned_project(database, principal, project_id)
    scheme = await project_point_scheme(database, project)
    filename = quote(f"{project['name']}-point-scheme.xlsx")
    return Response(
        point_scheme_xlsx(scheme),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )
