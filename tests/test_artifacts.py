from typing import Any, cast

import pytest
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.core.object_storage import StoredObject
from app.modules.agents import artifacts
from app.modules.agents.artifacts import artifact_tools, create_artifact, safe_file_name
from app.modules.agents.tools import ToolContext


class FakeArtifactCollection:
    def __init__(self) -> None:
        self.inserted: dict[str, Any] | None = None

    async def insert_one(self, document: dict[str, Any]) -> None:
        self.inserted = document


class FakeArtifactDatabase:
    def __init__(self) -> None:
        self.artifacts = FakeArtifactCollection()


class FakeArtifactStorage:
    def __init__(self) -> None:
        self.upload: dict[str, Any] | None = None

    async def put_bytes(self, **values: Any) -> StoredObject:
        self.upload = values
        return StoredObject("enerai", values["object_name"], "etag", None)

    async def delete_object(self, **_: Any) -> None:
        return None


def test_artifact_file_name_drops_paths_and_unsafe_characters() -> None:
    assert safe_file_name("../../reports/plant:summary?.csv") == "plant_summary_.csv"
    assert safe_file_name("冷站分析.md") == "冷站分析.md"


def test_empty_artifact_file_name_is_rejected() -> None:
    with pytest.raises(AppError) as captured:
        safe_file_name("../../")
    assert captured.value.code == "invalid_artifact_name"


def test_publish_artifact_has_explicit_write_policy() -> None:
    database = cast(AsyncDatabase[dict[str, Any]], object())
    tool = artifact_tools(database)[0]
    assert tool.name == "publish_artifact"
    assert tool.effect == "write"
    assert tool.execution_mode == "sequential"
    assert tool.idempotent is False
    assert tool.result_visibility == "both"


@pytest.mark.asyncio
async def test_artifact_is_written_to_minio(monkeypatch: pytest.MonkeyPatch) -> None:
    database = FakeArtifactDatabase()
    storage = FakeArtifactStorage()
    monkeypatch.setattr(artifacts, "get_minio_storage", lambda: storage)

    summary = await create_artifact(  # type: ignore[arg-type]
        database,
        ToolContext(
            run_id="run_test",
            session_id="ses_test",
            project_id="prj_test",
            user_id="usr_test",
        ),
        title="Report",
        file_name="report.md",
        media_type="text/markdown",
        content="# Report",
        presentation="preview-and-download",
    )

    assert summary.id.startswith("art_")
    assert storage.upload is not None
    assert storage.upload["object_name"].startswith(
        "artifacts/prj_test/ses_test/art_"
    )
    assert database.artifacts.inserted is not None
    assert database.artifacts.inserted["storage_bucket"] == "enerai"
    assert "file_id" not in database.artifacts.inserted
