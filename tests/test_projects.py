from __future__ import annotations

import pytest

from app.core.errors import AppError
from app.core.security import Principal
from app.modules.projects.schemas import ProjectCreate, ProjectUpdate
from app.modules.projects.service import (
    create_project,
    delete_project,
    get_project,
    list_projects,
    update_project,
)
from tests.fakes import InMemoryProjectRepository

OWNER = Principal(user_id="usr_owner", username="owner")
OTHER = Principal(user_id="usr_other", username="other")


async def test_project_lifecycle_is_owner_scoped() -> None:
    repository = InMemoryProjectRepository()
    created = await create_project(
        repository,
        OWNER,
        ProjectCreate(name=" Central Plant ", description=" Main campus "),
    )

    assert created.name == "Central Plant"
    assert created.description == "Main campus"
    assert (await list_projects(repository, OWNER)).total == 1
    assert (await list_projects(repository, OTHER)).total == 0

    updated = await update_project(
        repository,
        OWNER,
        created.id,
        ProjectUpdate(description="Updated"),
    )
    assert updated.description == "Updated"

    with pytest.raises(AppError) as hidden:
        await get_project(repository, OTHER, created.id)
    assert hidden.value.code == "project_not_found"

    await delete_project(repository, OWNER, created.id)
    assert (await list_projects(repository, OWNER)).total == 0


async def test_duplicate_project_name_is_rejected_case_insensitively() -> None:
    repository = InMemoryProjectRepository()
    await create_project(repository, OWNER, ProjectCreate(name="Plant"))

    with pytest.raises(AppError) as raised:
        await create_project(repository, OWNER, ProjectCreate(name="plant"))

    assert raised.value.code == "project_name_conflict"
