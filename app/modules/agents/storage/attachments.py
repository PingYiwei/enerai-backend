from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import UploadFile
from gridfs import AsyncGridFSBucket
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.ids import new_id
from app.core.object_storage import get_minio_storage
from app.core.security import Principal
from app.modules.agents.schemas import AttachmentSummary

Document = dict[str, Any]
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def detect_image_type(content: bytes) -> str | None:
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _summary(document: Document) -> AttachmentSummary:
    return AttachmentSummary(
        id=document["_id"],
        project_id=document["project_id"],
        name=document["name"],
        media_type=document["media_type"],
        size=document["size"],
        status=document["status"],
        created_at=document["created_at"],
    )


async def create_attachment(
    database: AsyncDatabase[Document],
    principal: Principal,
    project_id: str,
    file: UploadFile,
) -> AttachmentSummary:
    project = await database.projects.find_one(
        {"_id": project_id, "owner_id": principal.user_id}, {"_id": 1}
    )
    if project is None:
        raise AppError("project_not_found", "Project was not found", status_code=404)
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if not content:
        raise AppError("empty_attachment", "Image file is empty", status_code=422)
    if len(content) > MAX_IMAGE_BYTES:
        raise AppError("attachment_too_large", "Image exceeds the 10 MB limit", status_code=413)
    media_type = detect_image_type(content)
    if media_type is None:
        raise AppError(
            "unsupported_image",
            "Only PNG, JPEG, WebP, and GIF images are supported",
            status_code=422,
        )
    attachment_id = new_id("att")
    name = (file.filename or "image").strip()
    extension = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
    }[media_type]
    object_name = f"chat-images/{project_id}/{attachment_id}.{extension}"
    storage = get_minio_storage()
    stored = await storage.put_bytes(
        object_name=object_name,
        content=content,
        content_type=media_type,
        metadata={
            "attachment_id": attachment_id,
            "owner_id": principal.user_id,
            "project_id": project_id,
        },
    )
    document = {
        "_id": attachment_id,
        "project_id": project_id,
        "owner_id": principal.user_id,
        "name": name,
        "media_type": media_type,
        "size": len(content),
        "status": "draft",
        "storage_bucket": stored.bucket,
        "object_name": stored.object_name,
        "etag": stored.etag,
        "version_id": stored.version_id,
        "created_at": datetime.now(UTC),
    }
    try:
        await database.chat_attachments.insert_one(document)
    except Exception:
        await storage.delete_object(bucket=stored.bucket, object_name=stored.object_name)
        raise
    return _summary(document)


async def read_attachment(
    database: AsyncDatabase[Document], principal: Principal, attachment_id: str
) -> tuple[AttachmentSummary, bytes]:
    document = await database.chat_attachments.find_one(
        {"_id": attachment_id, "owner_id": principal.user_id}
    )
    if document is None:
        raise AppError("attachment_not_found", "Attachment was not found", status_code=404)
    if document.get("object_name"):
        content = await get_minio_storage().get_bytes(
            bucket=document.get("storage_bucket"),
            object_name=document["object_name"],
        )
    else:
        # Compatibility for images uploaded before MinIO storage was enabled.
        bucket = AsyncGridFSBucket(database, bucket_name="chat_files")
        stream = await bucket.open_download_stream(document["file_id"])
        content = await stream.read()
    return _summary(document), content


async def delete_draft_attachment(
    database: AsyncDatabase[Document], principal: Principal, attachment_id: str
) -> None:
    document = await database.chat_attachments.find_one_and_delete(
        {"_id": attachment_id, "owner_id": principal.user_id, "status": "draft"}
    )
    if document is None:
        raise AppError(
            "attachment_not_found",
            "Draft attachment was not found",
            status_code=404,
        )
    if document.get("object_name"):
        await get_minio_storage().delete_object(
            bucket=document.get("storage_bucket"),
            object_name=document["object_name"],
        )
    else:
        bucket = AsyncGridFSBucket(database, bucket_name="chat_files")
        await bucket.delete(document["file_id"])


async def cleanup_expired_drafts(database: AsyncDatabase[Document]) -> int:
    expired = await database.chat_attachments.find(
        {
            "status": "draft",
            "created_at": {"$lt": datetime.now(UTC) - timedelta(hours=24)},
        }
    ).to_list(None)
    if not expired:
        return 0
    removed = 0
    for candidate in expired:
        document = await database.chat_attachments.find_one_and_delete(
            {"_id": candidate["_id"], "status": "draft"}
        )
        if document is None:
            continue
        if document.get("object_name"):
            await get_minio_storage().delete_object(
                bucket=document.get("storage_bucket"),
                object_name=document["object_name"],
            )
        else:
            bucket = AsyncGridFSBucket(database, bucket_name="chat_files")
            await bucket.delete(document["file_id"])
        removed += 1
    return removed
