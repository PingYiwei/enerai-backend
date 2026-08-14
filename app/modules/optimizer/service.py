from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import UploadFile
from gridfs import AsyncGridFSBucket
from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.security import Principal
from app.modules.optimizer.schemas import (
    DatasetList,
    DatasetPreview,
    DatasetSummary,
    DeviceType,
)
from app.modules.optimizer.validation import count_valid_rows, decode_csv, validate_dataset

Document = dict[str, Any]
MAX_DATASET_BYTES = 20 * 1024 * 1024


async def _owned_project(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> None:
    project = await database.projects.find_one(
        {"_id": project_id, "owner_id": principal.user_id}, {"_id": 1}
    )
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)


def _summary(document: Document) -> DatasetSummary:
    payload = dict(document)
    payload.setdefault("description", "")
    payload.setdefault("file_size", 0)
    payload.setdefault(
        "valid_row_count",
        int(payload.get("row_count", 0)) if payload.get("status") == "valid" else 0,
    )
    return DatasetSummary.model_validate(payload)


async def create_dataset(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    name: str,
    description: str,
    device_type: DeviceType,
    file: UploadFile,
) -> DatasetSummary:
    await _owned_project(database, principal, project_id)
    filename = (file.filename or "dataset.csv").strip()
    if not filename.lower().endswith(".csv"):
        raise AppError("invalid_dataset_file", "Only CSV datasets are supported", status_code=422)
    content = await file.read(MAX_DATASET_BYTES + 1)
    if not content:
        raise AppError("empty_dataset", "Dataset file is empty", status_code=422)
    if len(content) > MAX_DATASET_BYTES:
        raise AppError("dataset_too_large", "Dataset exceeds the 20 MB limit", status_code=413)
    try:
        columns, rows = decode_csv(content)
    except ValueError as error:
        raise AppError("invalid_csv", str(error), status_code=422) from error
    validation = validate_dataset(device_type, columns, rows)
    valid_row_count = count_valid_rows(device_type, columns, rows)
    now = datetime.now(UTC)
    dataset_id = new_id("dts")
    bucket = AsyncGridFSBucket(database, bucket_name="optimizer_files")
    file_id = await bucket.upload_from_stream(
        filename,
        content,
        metadata={
            "owner_id": principal.user_id,
            "project_id": project_id,
            "dataset_id": dataset_id,
        },
    )
    document = {
        "_id": dataset_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        "name": name.strip(),
        "description": description.strip(),
        "filename": filename,
        "device_type": device_type,
        "status": (
            "valid"
            if all(rule.passed for rule in validation if rule.severity == "error")
            else "invalid"
        ),
        "row_count": len(rows),
        "valid_row_count": valid_row_count,
        "file_size": len(content),
        "columns": columns,
        "validation": [rule.model_dump(mode="json") for rule in validation],
        "file_id": file_id,
        "created_at": now,
    }
    await database.datasets.insert_one(document)
    return _summary(document)


async def list_datasets(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> DatasetList:
    await _owned_project(database, principal, project_id)
    documents = await (
        database.datasets.find({"project_id": project_id, "owner_id": principal.user_id})
        .sort("created_at", DESCENDING)
        .to_list(None)
    )
    return DatasetList(items=[_summary(document) for document in documents], total=len(documents))


async def owned_dataset(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    dataset_id: str,
) -> Document:
    document = await database.datasets.find_one(
        {
            "_id": dataset_id,
            "project_id": project_id,
            "owner_id": principal.user_id,
        }
    )
    if document is None:
        raise AppError("dataset_not_found", "Dataset was not found", status_code=404)
    return document


async def preview_dataset(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    dataset_id: str,
    *,
    offset: int,
    limit: int,
) -> DatasetPreview:
    document, content = await read_dataset_file(database, principal, project_id, dataset_id)
    columns, rows = decode_csv(content)
    selected = rows[offset : offset + limit]
    return DatasetPreview(
        columns=columns,
        rows=[
            {column: row[index] if index < len(row) else "" for index, column in enumerate(columns)}
            for row in selected
        ],
        total=int(document["row_count"]),
        offset=offset,
        limit=limit,
    )


async def read_dataset_file(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    dataset_id: str,
) -> tuple[Document, bytes]:
    document = await owned_dataset(database, principal, project_id, dataset_id)
    bucket = AsyncGridFSBucket(database, bucket_name="optimizer_files")
    stream = await bucket.open_download_stream(document["file_id"])
    content = await stream.read()
    return document, content


async def delete_dataset(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    dataset_id: str,
) -> None:
    document = await owned_dataset(database, principal, project_id, dataset_id)
    model_count = await database.models.count_documents({"dataset_id": dataset_id})
    if model_count:
        raise AppError(
            "dataset_in_use",
            "Delete models trained from this dataset first",
            status_code=409,
        )
    result = await database.datasets.delete_one({"_id": dataset_id})
    if result.deleted_count:
        bucket = AsyncGridFSBucket(database, bucket_name="optimizer_files")
        await bucket.delete(document["file_id"])
