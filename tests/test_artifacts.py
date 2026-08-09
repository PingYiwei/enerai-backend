from typing import Any, cast

import pytest
from pymongo.asynchronous.database import AsyncDatabase

from app.core.errors import AppError
from app.modules.agents.artifacts import artifact_tools, safe_file_name


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
