from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.optimizer.modeling import MODEL_FIELDS, train_model
from app.modules.optimizer.schemas import (
    ModelCreate,
    ModelList,
    ModelPreview,
    ModelSummary,
)
from app.modules.optimizer.service import owned_dataset, read_dataset_file
from app.modules.optimizer.validation import decode_csv

Document = dict[str, Any]


def _summary(document: Document) -> ModelSummary:
    return ModelSummary.model_validate(document)


async def create_model(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: ModelCreate,
) -> ModelSummary:
    dataset = await owned_dataset(database, principal, project_id, request.dataset_id)
    if dataset["status"] != "valid":
        raise AppError("dataset_invalid", "Only a valid dataset can train a model", status_code=409)
    _, content = await read_dataset_file(database, principal, project_id, request.dataset_id)
    columns, raw_rows = decode_csv(content)
    field_names, target_name = MODEL_FIELDS[dataset["device_type"]]
    indexes = {name: columns.index(name) for name in (*field_names, target_name)}
    rows = [
        {name: float(row[index]) for name, index in indexes.items()} for row in raw_rows[:50_000]
    ]
    try:
        result = await asyncio.to_thread(
            train_model, request.algorithm, dataset["device_type"], rows
        )
    except ValueError as error:
        raise AppError("model_training_failed", str(error), status_code=422) from error
    now = datetime.now(UTC)
    model_id = new_id("mdl")
    preview_rows = [
        {"actual": rows[index][target_name], "predicted": prediction}
        for index, prediction in enumerate(result.predictions[:500])
    ]
    document = {
        "_id": model_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        "dataset_id": request.dataset_id,
        "name": request.name.strip(),
        "device_type": dataset["device_type"],
        "algorithm": request.algorithm,
        "status": "ready",
        "metrics": result.metrics,
        "artifact": result.artifact,
        "preview": preview_rows,
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


async def preview_model(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    model_id: str,
) -> ModelPreview:
    document = await database.models.find_one(
        {"_id": model_id, "project_id": project_id, "owner_id": principal.user_id}
    )
    if document is None:
        raise AppError("model_not_found", "Model was not found", status_code=404)
    return ModelPreview(
        model=_summary(document),
        points=document.get("preview", []),
        artifact=document.get("artifact", {}),
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
