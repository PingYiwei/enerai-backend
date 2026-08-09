from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_principal, get_project_repository
from app.core.config import Settings
from app.core.security import Principal
from app.main import create_app
from tests.fakes import InMemoryProjectRepository


async def test_project_http_contract() -> None:
    repository = InMemoryProjectRepository()
    app = create_app(Settings(environment="test", jwt_secret="test-secret-with-at-least-32-bytes"))
    app.dependency_overrides[get_project_repository] = lambda: repository
    app.dependency_overrides[get_principal] = lambda: Principal(user_id="usr_test", username="test")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post(
            "/api/v1/projects",
            json={"name": "Plant", "description": "Test project"},
        )
        listed = await client.get("/api/v1/projects")
        fetched = await client.get(f"/api/v1/projects/{created.json()['id']}")

    assert created.status_code == 201
    assert listed.json()["total"] == 1
    assert fetched.json()["name"] == "Plant"
