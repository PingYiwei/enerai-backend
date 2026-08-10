from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from io import BytesIO
from typing import Any, cast

from minio import Minio
from minio.error import S3Error
from minio.helpers import DictType

from app.core.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class StoredObject:
    bucket: str
    object_name: str
    etag: str
    version_id: str | None


class MinioStorage:
    """Small async wrapper around the synchronous MinIO client."""

    def __init__(self, settings: Settings) -> None:
        if settings.minio_access_key is None or settings.minio_secret_key is None:
            raise RuntimeError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be configured")
        self.client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket
        self.presigned_url_expiry = timedelta(minutes=settings.minio_presigned_url_minutes)

    def _ensure_bucket_sync(self, bucket: str) -> None:
        try:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
        except S3Error as error:
            if error.code not in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                raise

    async def put_bytes(
        self,
        *,
        object_name: str,
        content: bytes,
        content_type: str,
        metadata: DictType | None = None,
        bucket: str | None = None,
    ) -> StoredObject:
        target_bucket = bucket or self.bucket
        await asyncio.to_thread(self._ensure_bucket_sync, target_bucket)
        result = await asyncio.to_thread(
            self.client.put_object,
            target_bucket,
            object_name,
            BytesIO(content),
            len(content),
            content_type=content_type,
            metadata=metadata,
        )
        if result.etag is None:
            raise RuntimeError(f"MinIO upload did not return an ETag for {object_name}")
        return StoredObject(
            bucket=target_bucket,
            object_name=result.object_name,
            etag=result.etag,
            version_id=getattr(result, "version_id", None),
        )

    def _get_bytes_sync(self, bucket: str, object_name: str) -> bytes:
        response: Any = self.client.get_object(bucket, object_name)
        try:
            return cast(bytes, response.read())
        finally:
            response.close()
            response.release_conn()

    async def get_bytes(self, *, object_name: str, bucket: str | None = None) -> bytes:
        return await asyncio.to_thread(
            self._get_bytes_sync,
            bucket or self.bucket,
            object_name,
        )

    async def delete_object(self, *, object_name: str, bucket: str | None = None) -> None:
        await asyncio.to_thread(
            self.client.remove_object,
            bucket or self.bucket,
            object_name,
        )

    async def create_presigned_get_url(
        self,
        *,
        object_name: str,
        bucket: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.client.presigned_get_object,
            bucket or self.bucket,
            object_name,
            expires=self.presigned_url_expiry,
        )


@lru_cache
def get_minio_storage() -> MinioStorage:
    return MinioStorage(get_settings())
