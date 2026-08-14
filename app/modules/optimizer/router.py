from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status

from app.api.dependencies import CurrentPrincipal, Database
from app.modules.optimizer.dataset_configs import DATASET_CONFIGS
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
    ModelCreate,
    ModelList,
    ModelPredictRequest,
    ModelPredictResponse,
    ModelPreview,
    ModelSummary,
)
from app.modules.optimizer.service import (
    create_dataset,
    delete_dataset,
    list_datasets,
    preview_dataset,
    read_dataset_file,
)

router = APIRouter()


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
    return await predict_model_outputs(
        database, principal, project_id, model_id, body.inputs
    )


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_model(
    project_id: str,
    model_id: str,
    database: Database,
    principal: CurrentPrincipal,
) -> Response:
    await delete_model(database, principal, project_id, model_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
