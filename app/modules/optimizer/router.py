from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile, status

from app.api.dependencies import CurrentPrincipal, Database
from app.core.config import Settings, get_settings
from app.modules.optimizer.dataset_configs import DATASET_CONFIGS
from app.modules.optimizer.engineering import (
    engineering_config,
    infer_topologies_with_llm,
    update_engineering_config,
)
from app.modules.optimizer.models import (
    create_model,
    delete_model,
    list_models,
    predict_model_outputs,
    preview_model,
)
from app.modules.optimizer.schemas import (
    DatasetList,
    DatasetPreview,
    DatasetSummary,
    DeviceType,
    DeviceTypeList,
    DeviceTypeOption,
    EngineeringConfigUpdate,
    EngineeringConfigView,
    EngineeringTopologyInference,
    ModelCreate,
    ModelList,
    ModelPredictRequest,
    ModelPredictResponse,
    ModelPreview,
    ModelSummary,
    OptimizationPreflightResult,
    OptimizationRunView,
    OptimizationStrategyCreate,
    OptimizationStrategyList,
    OptimizationStrategySummary,
)
from app.modules.optimizer.service import (
    create_dataset,
    delete_dataset,
    list_datasets,
    preview_dataset,
    read_dataset_file,
)
from app.modules.optimizer.strategies import (
    create_strategy,
    delete_strategy,
    latest_run,
    list_strategies,
    preflight_strategy,
    read_run,
    start_run,
    update_strategy,
)

router = APIRouter()


@router.post(
    "/strategies", response_model=OptimizationStrategySummary, status_code=status.HTTP_201_CREATED
)
async def create_optimization_strategy(
    project_id: str,
    body: OptimizationStrategyCreate,
    database: Database,
    principal: CurrentPrincipal,
) -> OptimizationStrategySummary:
    return await create_strategy(database, principal, project_id, body)


@router.get("/strategies", response_model=OptimizationStrategyList)
async def optimization_strategies(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> OptimizationStrategyList:
    return await list_strategies(database, principal, project_id)


@router.put("/strategies/{strategy_id}", response_model=OptimizationStrategySummary)
async def save_optimization_strategy(
    project_id: str,
    strategy_id: str,
    body: OptimizationStrategyCreate,
    database: Database,
    principal: CurrentPrincipal,
) -> OptimizationStrategySummary:
    return await update_strategy(database, principal, project_id, strategy_id, body)


@router.delete("/strategies/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_optimization_strategy(
    project_id: str, strategy_id: str, database: Database, principal: CurrentPrincipal
) -> Response:
    await delete_strategy(database, principal, project_id, strategy_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/strategies/{strategy_id}/preflight", response_model=OptimizationPreflightResult)
async def preflight_optimization_strategy(
    project_id: str, strategy_id: str, database: Database, principal: CurrentPrincipal
) -> OptimizationPreflightResult:
    return await preflight_strategy(database, principal, project_id, strategy_id)


@router.post("/strategies/{strategy_id}/runs", response_model=OptimizationRunView)
async def run_optimization_strategy(
    project_id: str, strategy_id: str, database: Database, principal: CurrentPrincipal
) -> OptimizationRunView:
    return await start_run(database, principal, project_id, strategy_id)


@router.get("/runs/latest", response_model=OptimizationRunView | None)
async def latest_optimization_run(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> OptimizationRunView | None:
    return await latest_run(database, principal, project_id)


@router.get("/runs/{run_id}", response_model=OptimizationRunView)
async def optimization_run(
    project_id: str, run_id: str, database: Database, principal: CurrentPrincipal
) -> OptimizationRunView:
    return await read_run(database, principal, project_id, run_id)


@router.get("/engineering", response_model=EngineeringConfigView)
async def read_engineering_config(
    project_id: str, database: Database, principal: CurrentPrincipal
) -> EngineeringConfigView:
    return await engineering_config(database, principal, project_id)


@router.put("/engineering", response_model=EngineeringConfigView)
async def save_engineering_config(
    project_id: str,
    body: EngineeringConfigUpdate,
    database: Database,
    principal: CurrentPrincipal,
) -> EngineeringConfigView:
    return await update_engineering_config(database, principal, project_id, body)


@router.post("/engineering/infer/llm", response_model=EngineeringTopologyInference)
async def infer_engineering_topology_with_llm(
    project_id: str,
    request: Request,
    database: Database,
    principal: CurrentPrincipal,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EngineeringTopologyInference:
    providers = request.app.state.providers
    return await infer_topologies_with_llm(database, principal, project_id, settings, providers)


@router.post("/datasets", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: str,
    database: Database,
    principal: CurrentPrincipal,
    name: Annotated[str, Form(min_length=1, max_length=120)],
    device_type: Annotated[DeviceType, Form()],
    file: Annotated[UploadFile, File()],
    description: Annotated[str, Form(max_length=300)] = "",
) -> DatasetSummary:
    return await create_dataset(
        database, principal, project_id, name, description, device_type, file
    )


@router.get("/datasets/device-types", response_model=DeviceTypeList)
async def dataset_device_types(_principal: CurrentPrincipal) -> DeviceTypeList:
    return DeviceTypeList(
        items=[
            DeviceTypeOption(
                value=config.device_type,
                label=config.label,
                fields=list(config.fields),
            )
            for config in DATASET_CONFIGS.values()
        ]
    )


@router.get("/datasets", response_model=DatasetList)
async def datasets(project_id: str, database: Database, principal: CurrentPrincipal) -> DatasetList:
    return await list_datasets(database, principal, project_id)


@router.get("/datasets/{dataset_id}/preview", response_model=DatasetPreview)
async def dataset_preview(
    project_id: str,
    dataset_id: str,
    database: Database,
    principal: CurrentPrincipal,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> DatasetPreview:
    return await preview_dataset(
        database, principal, project_id, dataset_id, offset=offset, limit=limit
    )


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    project_id: str,
    dataset_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    document, content = await read_dataset_file(database, principal, project_id, dataset_id)
    filename = quote(str(document["filename"]))
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_dataset(
    project_id: str,
    dataset_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    await delete_dataset(database, principal, project_id, dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/models", response_model=ModelSummary, status_code=status.HTTP_201_CREATED)
async def train_model(
    project_id: str,
    body: ModelCreate,
    database: Database,
    principal: CurrentPrincipal,
) -> ModelSummary:
    return await create_model(database, principal, project_id, body)


@router.get("/models", response_model=ModelList)
async def models(project_id: str, database: Database, principal: CurrentPrincipal) -> ModelList:
    return await list_models(database, principal, project_id)


@router.get("/models/{model_id}", response_model=ModelPreview)
async def model_preview(
    project_id: str,
    model_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> ModelPreview:
    return await preview_model(database, principal, project_id, model_id)


@router.post("/models/{model_id}/predict", response_model=ModelPredictResponse)
async def model_predict(
    project_id: str,
    model_id: str,
    body: ModelPredictRequest,
    database: Database,
    principal: CurrentPrincipal,
) -> ModelPredictResponse:
    return await predict_model_outputs(database, principal, project_id, model_id, body.inputs)


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_model(
    project_id: str,
    model_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    await delete_model(database, principal, project_id, model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
