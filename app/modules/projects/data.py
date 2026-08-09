from __future__ import annotations

import csv
import io
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from gridfs import AsyncGridFSBucket
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.projects.schemas import (
    DataQuery,
    DataQueryResult,
    DataSourceTestResult,
    DataSourceUpdate,
    DataSourceView,
    PointScheme,
    PointSchemeItem,
    PropertyCatalog,
)

Document = dict[str, Any]


async def owned_project(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> Document:
    document = await database.projects.find_one({"_id": project_id, "owner_id": principal.user_id})
    if document is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    return document


def _validate_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if urlparse(url).scheme not in {"http", "https"}:
        raise AppError(
            "invalid_data_source_url",
            "Data source URL must use http or https",
            status_code=422,
        )
    return url


def _view(config: Document) -> DataSourceView:
    return DataSourceView(
        base_url=config["base_url"],
        properties_path=config["properties_path"],
        query_path=config["query_path"],
        token_present=bool(config.get("bearer_token")),
        updated_at=config["updated_at"],
    )


async def save_data_source(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: DataSourceUpdate,
) -> DataSourceView:
    project = await owned_project(database, principal, project_id)
    previous = project.get("data_source", {})
    token = request.bearer_token
    if token is None and isinstance(previous, dict):
        token = previous.get("bearer_token")
    config: Document = {
        "base_url": _validate_base_url(request.base_url),
        "properties_path": request.properties_path.strip(),
        "query_path": request.query_path.strip(),
        "bearer_token": token.strip() if isinstance(token, str) else None,
        "updated_at": datetime.now(UTC),
    }
    await database.projects.update_one(
        {"_id": project_id, "owner_id": principal.user_id},
        {"$set": {"data_source": config, "updated_at": config["updated_at"]}},
    )
    return _view(config)


async def get_data_source(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> DataSourceView:
    project = await owned_project(database, principal, project_id)
    return _view(_configured_source(project))


def _configured_source(project: Document) -> Document:
    source = project.get("data_source")
    if not isinstance(source, dict):
        raise AppError(
            "data_source_not_configured",
            "Configure the project data source first",
            status_code=409,
        )
    return source


def _headers(source: Document) -> dict[str, str]:
    token = source.get("bearer_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _endpoint(source: Document, path_key: str) -> str:
    return urljoin(f"{source['base_url'].rstrip('/')}/", str(source[path_key]).lstrip("/"))


async def test_data_source(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> DataSourceTestResult:
    project = await owned_project(database, principal, project_id)
    source = _configured_source(project)
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                _endpoint(source, "properties_path"), headers=_headers(source)
            )
    except httpx.HTTPError as error:
        raise AppError("data_source_unreachable", str(error), status_code=502) from error
    elapsed_ms = round((time.perf_counter() - started) * 1_000)
    return DataSourceTestResult(
        ok=response.is_success,
        status_code=response.status_code,
        elapsed_ms=elapsed_ms,
    )


async def properties(
    database: AsyncDatabase[Document], principal: Principal, project_id: str
) -> PropertyCatalog:
    project = await owned_project(database, principal, project_id)
    source = _configured_source(project)
    payload = await _request_json("GET", _endpoint(source, "properties_path"), _headers(source))
    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise AppError(
            "invalid_data_source_response",
            "Property endpoint must return a list or an object with an items list",
            status_code=502,
        )
    return PropertyCatalog(items=items, total=len(items))


async def query_data(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    request: DataQuery,
) -> DataQueryResult:
    if request.end <= request.start:
        raise AppError("invalid_time_range", "end must be after start", status_code=422)
    project = await owned_project(database, principal, project_id)
    source = _configured_source(project)
    payload = await _request_json(
        "POST",
        _endpoint(source, "query_path"),
        _headers(source),
        json={
            "property_ids": request.property_ids,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "limit": request.limit,
        },
    )
    return DataQueryResult(data=payload)


async def _request_json(method: str, url: str, headers: dict[str, str], **kwargs: Any) -> Any:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise AppError("data_source_request_failed", str(error), status_code=502) from error


def point_scheme(project: Document) -> PointScheme:
    items: list[PointSchemeItem] = []
    for node in project.get("nodes", []):
        data = node.get("data", {})
        if not isinstance(data, dict) or not data.get("property"):
            continue
        items.append(
            PointSchemeItem(
                node_id=str(node["id"]),
                node_name=str(data.get("label") or node["id"]),
                node_type=str(node.get("type", "equipment")),
                property=str(data["property"]),
                unit=str(data.get("unit", "")),
            )
        )
    return PointScheme(items=items, total=len(items))


def point_scheme_csv(scheme: PointScheme) -> bytes:
    target = io.StringIO(newline="")
    writer = csv.writer(target)
    writer.writerow(["node_id", "node_name", "node_type", "property", "unit"])
    for item in scheme.items:
        writer.writerow([item.node_id, item.node_name, item.node_type, item.property, item.unit])
    return target.getvalue().encode("utf-8-sig")


def project_rdf(project: Document) -> str:
    lines = [
        "@prefix nodex: <https://nodex.dev/schema#> .",
        "@prefix project: <https://nodex.dev/project/> .",
        "",
    ]
    project_ref = f"project:{project['_id']}"
    lines.append(f'{project_ref} a nodex:Project ; nodex:name "{_turtle(project["name"])}" .')
    for node in project.get("nodes", []):
        node_ref = f"project:{project['_id']}/node/{node['id']}"
        data = node.get("data", {}) if isinstance(node.get("data"), dict) else {}
        node_type = _turtle(node.get("type", "equipment"))
        lines.append(
            f'{node_ref} a nodex:Equipment ; nodex:type "{node_type}" ; '
            f'nodex:name "{_turtle(data.get("label", node["id"]))}" .'
        )
    for edge in project.get("edges", []):
        lines.append(
            f"project:{project['_id']}/node/{edge['source']} nodex:connectedTo "
            f"project:{project['_id']}/node/{edge['target']} ."
        )
    return "\n".join(lines) + "\n"


def _turtle(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


async def cleanup_project_resources(
    database: AsyncDatabase[Document], owner_id: str, project_id: str
) -> None:
    datasets = await database.datasets.find(
        {"project_id": project_id, "owner_id": owner_id}, {"file_id": 1}
    ).to_list(None)
    bucket = AsyncGridFSBucket(database, bucket_name="optimizer_files")
    for dataset in datasets:
        if dataset.get("file_id") is not None:
            await bucket.delete(dataset["file_id"])
    attachments = await database.chat_attachments.find(
        {"project_id": project_id, "owner_id": owner_id}, {"file_id": 1}
    ).to_list(None)
    chat_bucket = AsyncGridFSBucket(database, bucket_name="chat_files")
    for attachment in attachments:
        await chat_bucket.delete(attachment["file_id"])
    await database.chat_attachments.delete_many({"project_id": project_id, "owner_id": owner_id})
    artifacts = await database.artifacts.find(
        {"project_id": project_id, "owner_id": owner_id}, {"file_id": 1}
    ).to_list(None)
    artifact_bucket = AsyncGridFSBucket(database, bucket_name="agent_artifacts")
    for artifact in artifacts:
        await artifact_bucket.delete(artifact["file_id"])
    await database.artifacts.delete_many({"project_id": project_id, "owner_id": owner_id})

    sessions = await database.agent_sessions.find(
        {"project_id": project_id, "owner_id": owner_id}, {"_id": 1}
    ).to_list(None)
    session_ids = [document["_id"] for document in sessions]
    operations = await database.agent_operations.find(
        {"project_id": project_id, "owner_id": owner_id}, {"_id": 1}
    ).to_list(None)
    run_ids = [document["_id"] for document in operations]
    if run_ids:
        await database.agent_events.delete_many({"run_id": {"$in": run_ids}})
        await database.agent_records.delete_many({"operation_id": {"$in": run_ids}})
    if session_ids:
        await database.agent_entries.delete_many({"session_id": {"$in": session_ids}})
        await database.agent_lanes.delete_many({"session_id": {"$in": session_ids}})
    await database.agent_operations.delete_many({"project_id": project_id, "owner_id": owner_id})
    await database.agent_sessions.delete_many({"project_id": project_id, "owner_id": owner_id})
    for collection in (
        database.studio_graph_versions,
        database.inspection_policies,
        database.inspection_runs,
        database.models,
        database.datasets,
    ):
        await collection.delete_many({"project_id": project_id, "owner_id": owner_id})
