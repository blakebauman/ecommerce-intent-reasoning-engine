"""Integration tests for batch processing API.

Imports are delayed below to avoid pulling in the full app (and spacy) before mocks.
"""
# ruff: noqa: E402

import importlib.util
import os
import sys
from unittest.mock import AsyncMock

import pytest


# Helper to load modules directly
def _load_module_directly(module_name: str, file_path: str):
    """Load a module directly from file, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get base path
_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_api_path = os.path.join(_base_path, "src", "intent_engine", "api")

# Pre-load batch_models to prevent api/__init__.py from being triggered
_batch_models = _load_module_directly(
    "intent_engine.api.batch_models", os.path.join(_api_path, "batch_models.py")
)

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Import batch queue and tenancy directly (these don't have problematic imports)
from intent_engine.batch.queue import BatchJob, BatchQueue, JobStatus
from intent_engine.tenancy.models import TenantConfig, TenantTier

# Load batch router module directly
_batch_module = _load_module_directly(
    "intent_engine.api.batch_direct", os.path.join(_api_path, "batch.py")
)
batch_router = _batch_module.router
set_queue_override = _batch_module.set_queue_override
set_tenant_override = _batch_module.set_tenant_override


class TestBatchAPI:
    """Integration tests for batch API endpoints."""

    @pytest.fixture
    def mock_queue(self):
        """Create a mock batch queue."""
        queue = AsyncMock(spec=BatchQueue)
        return queue

    @pytest.fixture
    def tenant(self):
        """Create a test tenant."""
        return TenantConfig(
            tenant_id="batch-test",
            name="Batch Test",
            tier=TenantTier.PROFESSIONAL,
            api_key="batch-key",
            batch_processing_enabled=True,
        )

    @pytest.fixture(autouse=True)
    def cleanup_overrides(self):
        """Clean up overrides after each test."""
        yield
        set_queue_override(None)
        set_tenant_override(None)

    def test_create_email_batch(self, mock_queue, tenant):
        """Test creating an email batch job."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="new-job-123",
            tenant_id="batch-test",
            status=JobStatus.QUEUED,
            total_items=2,
        )
        mock_queue.enqueue.return_value = mock_job

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.post(
            "/v1/batch/emails",
            json={
                "items": [
                    {"subject": "Test 1", "body": "Body 1", "from_email": "a@b.com"},
                    {"subject": "Test 2", "body": "Body 2", "from_email": "c@d.com"},
                ],
                "priority": "normal",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "new-job-123"
        assert data["status"] == "queued"
        assert data["total_items"] == 2

    def test_create_batch_exceeds_limit(self, mock_queue, tenant):
        """Test creating batch that exceeds size limit."""
        app = FastAPI()
        app.include_router(batch_router)

        # Set a low limit
        tenant.max_batch_size = 2

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.post(
            "/v1/batch/emails",
            json={
                "items": [
                    {"subject": f"Test {i}", "body": "Body", "from_email": "a@b.com"}
                    for i in range(10)
                ],
            },
        )

        assert response.status_code == 400
        assert "exceeds limit" in response.json()["detail"]

    def test_create_batch_disabled(self, mock_queue):
        """Test creating batch when disabled for tenant."""
        app = FastAPI()
        app.include_router(batch_router)

        disabled_tenant = TenantConfig(
            tenant_id="disabled-batch",
            name="Disabled",
            tier=TenantTier.FREE,
            api_key="disabled-key",
            batch_processing_enabled=False,
        )

        set_queue_override(mock_queue)
        set_tenant_override(disabled_tenant)

        client = TestClient(app)
        response = client.post(
            "/v1/batch/emails",
            json={
                "items": [{"subject": "Test", "body": "Body", "from_email": "a@b.com"}],
            },
        )

        assert response.status_code == 403

    def test_get_job_status(self, mock_queue, tenant):
        """Test getting job status."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="job-456",
            tenant_id="batch-test",
            status=JobStatus.PROCESSING,
            total_items=100,
            processed_items=50,
        )
        mock_queue.get_job.return_value = mock_job

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.get("/v1/batch/jobs/job-456")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-456"
        assert data["status"] == "processing"
        assert data["progress"] == 0.5

    def test_get_job_not_found(self, mock_queue, tenant):
        """Test getting non-existent job."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_queue.get_job.return_value = None

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.get("/v1/batch/jobs/nonexistent")

        assert response.status_code == 404

    def test_get_job_wrong_tenant(self, mock_queue, tenant):
        """Test accessing job from different tenant."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="other-job",
            tenant_id="other-tenant",
            status=JobStatus.COMPLETED,
        )
        mock_queue.get_job.return_value = mock_job

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.get("/v1/batch/jobs/other-job")

        assert response.status_code == 404

    def test_get_job_results(self, mock_queue, tenant):
        """Test getting completed job results."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="completed-job",
            tenant_id="batch-test",
            status=JobStatus.COMPLETED,
            total_items=2,
            processed_items=2,
        )
        mock_queue.get_job.return_value = mock_job
        mock_queue.get_results.return_value = [
            {"item_id": "1", "success": True, "result": {"intent": "A"}},
            {"item_id": "2", "success": True, "result": {"intent": "B"}},
        ]

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.get("/v1/batch/jobs/completed-job/results")

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_get_results_incomplete_job(self, mock_queue, tenant):
        """Test getting results from incomplete job."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="processing-job",
            tenant_id="batch-test",
            status=JobStatus.PROCESSING,
        )
        mock_queue.get_job.return_value = mock_job

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.get("/v1/batch/jobs/processing-job/results")

        assert response.status_code == 400
        assert "not complete" in response.json()["detail"]

    def test_cancel_job(self, mock_queue, tenant):
        """Test canceling a queued job."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="cancel-job",
            tenant_id="batch-test",
            status=JobStatus.QUEUED,
        )
        mock_queue.get_job.return_value = mock_job
        mock_queue.cancel_job.return_value = True

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.delete("/v1/batch/jobs/cancel-job")

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancel_processing_job(self, mock_queue, tenant):
        """Test canceling a processing job fails."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_job = BatchJob(
            job_id="processing-job",
            tenant_id="batch-test",
            status=JobStatus.PROCESSING,
        )
        mock_queue.get_job.return_value = mock_job
        mock_queue.cancel_job.return_value = False

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.delete("/v1/batch/jobs/processing-job")

        assert response.status_code == 400

    def test_list_jobs(self, mock_queue, tenant):
        """Test listing jobs."""
        app = FastAPI()
        app.include_router(batch_router)

        mock_queue.list_jobs.return_value = [
            BatchJob(
                job_id="job-1", tenant_id="batch-test", status=JobStatus.QUEUED, total_items=10
            ),
            BatchJob(
                job_id="job-2", tenant_id="batch-test", status=JobStatus.COMPLETED, total_items=20
            ),
        ]

        set_queue_override(mock_queue)
        set_tenant_override(tenant)

        client = TestClient(app)
        response = client.get("/v1/batch/jobs")

        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
