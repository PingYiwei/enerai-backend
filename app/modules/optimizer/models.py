from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import Any, cast

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.optimizer.modeling import MODEL_FIELDS, predict_model, train_model
from app.modules.optimizer.schemas import (
    DeviceType,
    ModelCreate,
    ModelList,
    ModelPredictResponse,
    ModelPreview,
    ModelSeries,
    ModelSummary,
)
from app.modules.optimizer.service import owned_dataset, read_dataset_file
from app.modules.optimizer.validation import decode_csv

Document = dict[str, Any]


def _summary(document: Document) -> ModelSummary:
    payload = dict(document)
    payload.setdefault("description", "")
    payload.setdefault("dataset_name", "")
    payload.setdefault("usage_number", 0)
    return ModelSummary.model_validate(payload)


async def create_model(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: ModelCreate,
) -> ModelSummary:
    dataset = await owned_dataset(database, principal, project_id, request.dataset_id)
    if dataset["status"] != "valid":
        raise AppError("dataset_invalid", "Only a valid dataset can train a model", status_code=409)
    device_type = cast(DeviceType, dataset["device_type"])
    _, content = await read_dataset_file(database, principal, project_id, request.dataset_id)
    columns, raw_rows = decode_csv(content)
    required_fields = MODEL_FIELDS[device_type]
    try:
        indexes = {name: columns.index(name) for name in required_fields}
    except ValueError as error:
        raise AppError(
            "model_fields_missing",
            "Dataset does not contain the fields required for this device model",
            status_code=422,
        ) from error
    rows = [
        {name: float(row[index]) for name, index in indexes.items()}
        for row in raw_rows[:50_000]
        if len(row) == len(columns)
    ]
    if not all(math.isfinite(value) for row in rows for value in row.values()):
        raise AppError(
            "model_data_invalid",
            "Model training fields must contain finite numeric values",
            status_code=422,
        )
    try:
        result = await asyncio.to_thread(
            train_model,
            request.algorithm,
            device_type,
            rows,
        )
    except ValueError as error:
        raise AppError("model_training_failed", str(error), status_code=422) from error
    now = datetime.now(UTC)
    model_id = new_id("mdl")
    document: Document = {
        "_id": model_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        "dataset_id": request.dataset_id,
        "dataset_name": str(dataset.get("name", "")),
        "name": request.name.strip(),
        "description": request.description.strip(),
        "device_type": device_type,
        "algorithm": request.algorithm,
        "status": "ready",
        "metrics": result.metrics,
        "artifact": result.artifact,
        "series": result.series,
        "usage_number": 0,
        "created_at": now,
    }
    await database.models.insert_one(document)
    return _summary(document)


async def list_models(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> ModelList:
    documents = await (
        database.models.find({"project_id": project_id, "owner_id": principal.user_id})
        .sort("created_at", DESCENDING)
        .to_list(None)
    )
    return ModelList(items=[_summary(document) for document in documents], total=len(documents))


async def owned_model(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    model_id: str,
) -> Document:
    document = await database.models.find_one(
        {"_id": model_id, "project_id": project_id, "owner_id": principal.user_id}
    )
    if document is None:
        raise AppError("model_not_found", "Model was not found", status_code=404)
    return document


async def preview_model(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    model_id: str,
) -> ModelPreview:
    document = await owned_model(database, principal, project_id, model_id)
    raw_series = document.get("series", [])
    if not raw_series and document.get("preview"):
        artifact = document.get("artifact", {})
        raw_series = [{
            "key": "model",
            "name": "Performance model",
            "kind": "model",
            "input_fields": artifact.get("feature_names", []),
            "output_field": artifact.get("target_name", "output"),
            "metrics": document.get("metrics", {}),
            "formula": "",
            "points": document["preview"],
        }]
    return ModelPreview(
        model=_summary(document),
        series=[ModelSeries.model_validate(item) for item in raw_series],
        artifact=document.get("artifact", {}),
    )


async def predict_model_outputs(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    model_id: str,
    inputs: list[dict[str, float]],
) -> ModelPredictResponse:
    document = await owned_model(database, principal, project_id, model_id)
    device_type = cast(DeviceType, document["device_type"])
    try:
        outputs = [
            {
                name: round(value, 6)
                for name, value in predict_model(
                    device_type, document.get("artifact", {}), item
                ).items()
            }
            for item in inputs
        ]
    except ValueError as error:
        raise AppError("model_prediction_failed", str(error), status_code=422) from error
    await database.models.update_one({"_id": model_id}, {"$inc": {"usage_number": 1}})
    return ModelPredictResponse(
        outputs=outputs,
        usage_number=int(document.get("usage_number", 0)) + 1,
    )


async def delete_model(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    model_id: str,
) -> None:
    result = await database.models.delete_one(
        {"_id": model_id, "project_id": project_id, "owner_id": principal.user_id}
    )
    if not result.deleted_count:
        raise AppError("model_not_found", "Model was not found", status_code=404)
