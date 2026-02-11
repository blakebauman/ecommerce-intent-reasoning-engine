"""Integration tests for A2A protocol endpoints."""

import sys

import pytest

# Skip entire module on Python 3.14+ due to spaCy incompatibility
if sys.version_info >= (3, 14):
    pytest.skip(
        "spaCy not compatible with Python 3.14+ (uses Pydantic v1)",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient

from intent_engine.api.server import app
from intent_engine.config import get_settings


@pytest.fixture
def client() -> TestClient:
    """Create a test client (use context manager so lifespan runs and engine is initialized)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def api_key() -> str:
    """API key for auth on protected routes (tenant middleware)."""
    return get_settings().api_key


@pytest.fixture
def auth_headers(api_key: str) -> dict:
    """Headers with Bearer token for protected A2A routes."""
    return {"Authorization": f"Bearer {api_key}"}


class TestAgentCard:
    """Tests for the agent card endpoint."""

    def test_get_agent_card(self, client: TestClient) -> None:
        """Test fetching the agent card."""
        response = client.get("/.well-known/agent.json")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "intent-engine"
        assert data["version"] == "0.1.0"
        assert "capabilities" in data
        assert "actions" in data

    def test_agent_card_capabilities(self, client: TestClient) -> None:
        """Test that agent card includes expected capabilities."""
        response = client.get("/.well-known/agent.json")
        data = response.json()

        assert "intent-resolution" in data["capabilities"]
        assert "entity-extraction" in data["capabilities"]
        assert "ecommerce" in data["capabilities"]

    def test_agent_card_actions(self, client: TestClient) -> None:
        """Test that agent card includes expected actions."""
        response = client.get("/.well-known/agent.json")
        data = response.json()

        action_names = [a["name"] for a in data["actions"]]
        assert "resolve_intent" in action_names
        assert "classify_intent_fast" in action_names
        assert "list_intent_taxonomy" in action_names


class TestA2ATaskSubmission:
    """Tests for A2A task submission (requires engine initialization)."""

    def test_submit_list_taxonomy_sync(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test synchronous task submission for list_intent_taxonomy."""
        response = client.post(
            "/a2a/tasks",
            json={
                "action": "list_intent_taxonomy",
                "input": {},
                "async_mode": False,
            },
            headers=auth_headers,
        )
        # May return 503 if engine not initialized
        if response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["intent_count"] == 8

    def test_submit_unknown_action(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test that unknown actions return 400."""
        response = client.post(
            "/a2a/tasks",
            json={
                "action": "unknown_action",
                "input": {},
            },
            headers=auth_headers,
        )
        # May return 503 if engine not initialized
        if response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        assert response.status_code == 400
        assert "unknown action" in response.json()["detail"].lower()

    def test_submit_async_task(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test async task submission returns pending status."""
        response = client.post(
            "/a2a/tasks",
            json={
                "action": "list_intent_taxonomy",
                "input": {},
                "async_mode": True,
            },
            headers=auth_headers,
        )
        # May return 503 if engine not initialized
        if response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        # Async tasks may complete immediately for fast operations
        assert data["status"] in ["pending", "running", "completed"]


class TestA2ATaskStatus:
    """Tests for A2A task status endpoint (requires engine)."""

    def test_get_task_status(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test getting task status after submission."""
        # Submit a task first
        submit_response = client.post(
            "/a2a/tasks",
            json={
                "action": "list_intent_taxonomy",
                "input": {},
                "async_mode": False,
            },
            headers=auth_headers,
        )
        # May return 503 if engine not initialized
        if submit_response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        task_id = submit_response.json()["id"]

        # Get task status
        status_response = client.get(
            f"/a2a/tasks/{task_id}", headers=auth_headers
        )
        assert status_response.status_code == 200

        data = status_response.json()
        assert data["id"] == task_id
        assert data["status"] == "completed"

    def test_get_nonexistent_task(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test that nonexistent task returns 404."""
        # First try to access any endpoint to trigger engine init
        _ = client.get("/.well-known/agent.json")

        response = client.get(
            "/a2a/tasks/nonexistent-task-id", headers=auth_headers
        )
        # May return 503 if engine not initialized
        if response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        assert response.status_code == 404


class TestA2ATaskCancellation:
    """Tests for A2A task cancellation (requires engine)."""

    def test_cancel_completed_task(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test cancelling a completed task returns not_cancelled."""
        # Submit and complete a task
        submit_response = client.post(
            "/a2a/tasks",
            json={
                "action": "list_intent_taxonomy",
                "input": {},
                "async_mode": False,
            },
            headers=auth_headers,
        )
        # May return 503 if engine not initialized
        if submit_response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        task_id = submit_response.json()["id"]

        # Try to cancel
        cancel_response = client.post(
            f"/a2a/tasks/{task_id}/cancel", headers=auth_headers
        )
        assert cancel_response.status_code == 200

        data = cancel_response.json()
        assert data["status"] == "not_cancelled"
        assert "completed" in data["reason"]

    def test_cancel_nonexistent_task(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test that cancelling nonexistent task returns 404."""
        # First try to access any endpoint to trigger engine init
        _ = client.get("/.well-known/agent.json")

        response = client.post(
            "/a2a/tasks/nonexistent-task-id/cancel", headers=auth_headers
        )
        # May return 503 if engine not initialized
        if response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        assert response.status_code == 404


class TestA2AResolveIntent:
    """Tests for resolve_intent action via A2A (requires database)."""

    @pytest.mark.skip(reason="Requires database connection")
    def test_resolve_intent_sync(self, client: TestClient) -> None:
        """Test resolve_intent action."""
        response = client.post(
            "/a2a/tasks",
            json={
                "action": "resolve_intent",
                "input": {
                    "raw_text": "Where is my order #12345?",
                },
                "async_mode": False,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        assert data["result"]["resolved_intents"] is not None

    def test_resolve_intent_missing_text(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        """Test resolve_intent fails without raw_text."""
        response = client.post(
            "/a2a/tasks",
            json={
                "action": "resolve_intent",
                "input": {},  # Missing raw_text
                "async_mode": False,
            },
            headers=auth_headers,
        )
        # May return 503 if engine not initialized
        if response.status_code == 503:
            pytest.skip("Engine not initialized - requires database")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "failed"
        assert "raw_text" in data["error"].lower()
