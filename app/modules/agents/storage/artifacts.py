from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from gridfs import AsyncGridFSBucket
from pymongo import DESCENDING
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.object_storage import get_minio_storage
from app.core.security import Principal
from app.modules.agents.runtime.types import JsonObject, ToolResult
from app.modules.agents.schemas import ArtifactList, ArtifactSummary
from app.modules.agents.tools.base import Tool, ToolContext

Document = dict[str, Any]
MAX_ARTIFACT_BYTES = 5 * 1024 * 1024
SUPPORTED_MEDIA_TYPES = {
    "application/json",
    "text/csv",
    "text/markdown",
    "text/plain",
}
_UNSAFE_FILENAME = re.compile(r"[^\w.()\- ]+", re.UNICODE)


def safe_file_name(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1].strip().strip(".")
    name = _UNSAFE_FILENAME.sub("_", name)[:160].strip()
    if not name:
        raise AppError("invalid_artifact_name", "Artifact file name is invalid", status_code=422)
    return name


def _summary(document: Document) -> ArtifactSummary:
    return ArtifactSummary(
        id=document["_id"],
        project_id=document["project_id"],
        session_id=document["session_id"],
        run_id=document["run_id"],
        title=document["title"],
        file_name=document["file_name"],
        media_type=document["media_type"],
        size=document["size"],
        presentation=document["presentation"],
        created_at=document["created_at"],
    )


async def create_artifact(
    database: AsyncDatabase[Document],
    context: ToolContext,
    *,
    title: str,
    file_name: str,
    media_type: str,
    content: str,
    presentation: str,
) -> ArtifactSummary:
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise AppError(
            "unsupported_artifact_type",
            "Artifacts must be plain text, Markdown, CSV, or JSON",
            status_code=422,
        )
    payload = content.encode("utf-8")
    if not payload:
        raise AppError("empty_artifact", "Artifact content is empty", status_code=422)
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise AppError("artifact_too_large", "Artifact exceeds the 5 MB limit", status_code=413)
    if media_type == "application/json":
        try:
            json.loads(content)
        except json.JSONDecodeError as error:
            raise AppError(
                "invalid_artifact_json",
                "JSON artifact content is invalid",
                status_code=422,
            ) from error

    artifact_id = new_id("art")
    stored_name = safe_file_name(file_name)
    object_name = (
        f"artifacts/{context.project_id}/{context.session_id}/{artifact_id}/{stored_name}"
    )
    storage = get_minio_storage()
    stored = await storage.put_bytes(
        object_name=object_name,
        content=payload,
        content_type=media_type,
        metadata={
            "artifact_id": artifact_id,
            "owner_id": context.user_id,
            "project_id": context.project_id,
            "session_id": context.session_id,
            "run_id": context.run_id,
        },
    )
    document: Document = {
        "_id": artifact_id,
        "owner_id": context.user_id,
        "project_id": context.project_id,
        "session_id": context.session_id,
        "run_id": context.run_id,
        "title": title.strip()[:160] or stored_name,
        "file_name": stored_name,
        "media_type": media_type,
        "size": len(payload),
        "presentation": presentation,
        "storage_bucket": stored.bucket,
        "object_name": stored.object_name,
        "etag": stored.etag,
        "version_id": stored.version_id,
        "created_at": datetime.now(UTC),
    }
    try:
        await database.artifacts.insert_one(document)
    except Exception:
        await storage.delete_object(bucket=stored.bucket, object_name=stored.object_name)
        raise
    return _summary(document)


async def list_artifacts(
    database: AsyncDatabase[Document], principal: Principal, session_id: str
) -> ArtifactList:
    session = await database.agent_sessions.find_one(
        {"_id": session_id, "owner_id": principal.user_id}, {"_id": 1}
    )
    if session is None:
        raise AppError("session_not_found", "Session was not found", status_code=404)
    documents = await (
        database.artifacts.find({"session_id": session_id, "owner_id": principal.user_id})
        .sort("created_at", DESCENDING)
        .to_list(None)
    )
    items = [_summary(document) for document in documents]
    return ArtifactList(items=items, total=len(items))


async def read_artifact(
    database: AsyncDatabase[Document], principal: Principal, artifact_id: str
) -> tuple[ArtifactSummary, bytes]:
    document = await database.artifacts.find_one(
        {"_id": artifact_id, "owner_id": principal.user_id}
    )
    if document is None:
        raise AppError("artifact_not_found", "Artifact was not found", status_code=404)
    if document.get("object_name"):
        content = await get_minio_storage().get_bytes(
            bucket=document.get("storage_bucket"),
            object_name=document["object_name"],
        )
    else:
        # Compatibility for artifacts created before MinIO storage was enabled.
        bucket = AsyncGridFSBucket(database, bucket_name="agent_artifacts")
        stream = await bucket.open_download_stream(document["file_id"])
        content = await stream.read()
    return _summary(document), content


def artifact_tools(database: AsyncDatabase[Document]) -> tuple[Tool, ...]:
    async def publish(arguments: JsonObject, context: ToolContext) -> ToolResult:
        artifact = await create_artifact(
            database,
            context,
            title=str(arguments["title"]),
            file_name=str(arguments["file_name"]),
            media_type=str(arguments["media_type"]),
            content=str(arguments["content"]),
            presentation=str(arguments.get("presentation", "preview-and-download")),
        )
        details = {"artifact": artifact.model_dump(mode="json")}
        return ToolResult(
            tool_call_id="",
            content=json.dumps(
                {
                    "artifact_id": artifact.id,
                    "title": artifact.title,
                    "media_type": artifact.media_type,
                    "size": artifact.size,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            details=details,
        )

    return (
        Tool(
            name="publish_artifact",
            description=(
                "Publish a final text, Markdown, CSV, or JSON deliverable for the user. "
                "Do not publish scratch work or duplicate the answer."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 160},
                    "file_name": {"type": "string", "minLength": 1, "maxLength": 160},
                    "media_type": {"type": "string", "enum": sorted(SUPPORTED_MEDIA_TYPES)},
                    "content": {"type": "string", "minLength": 1},
                    "presentation": {
                        "type": "string",
                        "enum": ["download-only", "preview-only", "preview-and-download"],
                        "default": "preview-and-download",
                    },
                },
                "required": ["title", "file_name", "media_type", "content"],
                "additionalProperties": False,
            },
            execute=publish,
            effect="write",
            execution_mode="sequential",
            result_visibility="both",
            idempotent=False,
        ),
    )
