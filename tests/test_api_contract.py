from app.core.config import Settings
from app.main import create_app


def test_frontend_critical_routes_are_present_in_openapi() -> None:
    app = create_app(Settings(environment="test", jwt_secret="test-secret-with-at-least-32-bytes"))
    paths = app.openapi()["paths"]
    expected = {
        "/api/v1/auth/token": "post",
        "/api/v1/projects": "get",
        "/api/v1/projects/{project_id}/token-usage": "get",
        "/api/v1/projects/{project_id}/studio/graph": "put",
        "/api/v1/projects/{project_id}/insight/attachments": "post",
        "/api/v1/projects/{project_id}/insight/context-options": "get",
        "/api/v1/projects/{project_id}/insight/sessions": "get",
        "/api/v1/sessions/{session_id}": "get",
        "/api/v1/sessions/{session_id}/runs": "post",
        "/api/v1/runs/{run_id}/events": "get",
        "/api/v1/sessions/{session_id}/artifacts": "get",
        "/api/v1/artifacts/{artifact_id}/content": "get",
        "/api/v1/projects/{project_id}/inspections/runs": "post",
        "/api/v1/projects/{project_id}/optimizer/datasets": "post",
        "/api/v1/projects/{project_id}/optimizer/models": "post",
    }
    for path, method in expected.items():
        assert method in paths[path]
    assert "patch" in paths["/api/v1/sessions/{session_id}"]
    assert "delete" in paths["/api/v1/sessions/{session_id}"]
