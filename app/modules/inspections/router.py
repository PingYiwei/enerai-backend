from fastapi import APIRouter, status

from app.api.dependencies import CurrentPrincipal, Database
from app.modules.inspections.schemas import (
    InspectionPolicy,
    InspectionPolicyUpdate,
    InspectionRun,
    InspectionRunList,
)
from app.modules.inspections.service import create_run, get_policy, list_runs, save_policy

router = APIRouter()


@router.get("/policy", response_model=InspectionPolicy)
async def policy(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionPolicy:
    return await get_policy(database, principal, project_id)


@router.put("/policy", response_model=InspectionPolicy)
async def update_policy(
    project_id: str,
    body: InspectionPolicyUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> InspectionPolicy:
    return await save_policy(database, principal, project_id, body)


@router.post("/runs", response_model=InspectionRun, status_code=status.HTTP_201_CREATED)
async def run_now(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionRun:
    return await create_run(database, principal, project_id)


@router.get("/runs", response_model=InspectionRunList)
async def runs(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> InspectionRunList:
    return await list_runs(database, principal, project_id)
