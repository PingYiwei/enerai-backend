from io import BytesIO
from typing import Any

import pytest
from fastapi import UploadFile

from app.core.object_storage import StoredObject
from app.core.security import Principal
from app.modules.agents.storage import attachments
from app.modules.agents.storage.attachments import create_attachment, detect_image_type


class FakeCollection:
    def __init__(self, document: dict[str, Any] | None = None) -> None:
        self.document = document
        self.inserted: dict[str, Any] | None = None

    async def find_one(self, *_: Any, **__: Any) -> dict[str, Any] | None:
        return self.document

    async def insert_one(self, document: dict[str, Any]) -> None:
        self.inserted = document


class FakeDatabase:
    def __init__(self) -> None:
        self.projects = FakeCollection({"_id": "prj_test"})
        self.chat_attachments = FakeCollection()


class FakeStorage:
    def __init__(self) -> None:
        self.upload: dict[str, Any] | None = None

    async def put_bytes(self, **values: Any) -> StoredObject:
        self.upload = values
        return StoredObject("enerai", values["object_name"], "etag", None)

    async def delete_object(self, **_: Any) -> None:
        return None


def test_image_type_is_detected_from_bytes_not_request_header() -> None:
    assert detect_image_type(b"\x89PNG\r\n\x1a\ncontent") == "image/png"
    assert detect_image_type(b"\xff\xd8\xffcontent") == "image/jpeg"
    assert detect_image_type(b"RIFF1234WEBPcontent") == "image/webp"
    assert detect_image_type(b"not-an-image") is None


@pytest.mark.asyncio
async def test_attachment_is_written_to_minio(monkeypatch: pytest.MonkeyPatch) -> None:
    database = FakeDatabase()
    storage = FakeStorage()
    monkeypatch.setattr(attachments, "get_minio_storage", lambda: storage)
    file = UploadFile(filename="plant.png", file=BytesIO(b"\x89PNG\r\n\x1a\ncontent"))

    summary = await create_attachment(  # type: ignore[arg-type]
        database,
        Principal(user_id="usr_test", username="test"),
        "prj_test",
        file,
    )

    assert summary.id.startswith("att_")
    assert storage.upload is not None
    assert storage.upload["object_name"].startswith("chat-images/prj_test/att_")
    assert storage.upload["content_type"] == "image/png"
    assert storage.upload["metadata"] == {
        "attachment-id": summary.id,
        "owner-id": "usr_test",
        "project-id": "prj_test",
    }
    assert database.chat_attachments.inserted is not None
    assert database.chat_attachments.inserted["storage_bucket"] == "enerai"
    assert "file_id" not in database.chat_attachments.inserted
