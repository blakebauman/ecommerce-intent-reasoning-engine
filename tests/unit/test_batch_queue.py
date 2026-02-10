"""Unit tests for batch processing queue."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

from intent_engine.batch.queue import BatchQueue, BatchJob, BatchJobItem, JobStatus, JobPriority


class TestBatchJob:
    """Tests for BatchJob model."""

    def test_progress_empty(self):
        """Test progress calculation with no items."""
        job = BatchJob(
            job_id="test-job",
            tenant_id="tenant-1",
            total_items=0,
        )
        assert job.progress() == 0.0

    def test_progress_partial(self):
        """Test progress calculation with partial completion."""
        job = BatchJob(
            job_id="test-job",
            tenant_id="tenant-1",
            total_items=100,
            processed_items=25,
        )
        assert job.progress() == 0.25

    def test_progress_complete(self):
        """Test progress calculation when complete."""
        job = BatchJob(
            job_id="test-job",
            tenant_id="tenant-1",
            total_items=50,
            processed_items=50,
        )
        assert job.progress() == 1.0

    def test_duration_not_started(self):
        """Test duration when not started."""
        job = BatchJob(
            job_id="test-job",
            tenant_id="tenant-1",
        )
        assert job.duration_seconds() is None

    def test_duration_in_progress(self):
        """Test duration while processing."""
        job = BatchJob(
            job_id="test-job",
            tenant_id="tenant-1",
            started_at=datetime.utcnow(),
        )
        duration = job.duration_seconds()
        assert duration is not None
        assert duration >= 0

    def test_default_status(self):
        """Test default job status."""
        job = BatchJob(
            job_id="test-job",
            tenant_id="tenant-1",
        )
        assert job.status == JobStatus.QUEUED


class TestBatchQueue:
    """Tests for BatchQueue class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        return redis

    @pytest.fixture
    def queue(self, mock_redis):
        """Create a batch queue with mock Redis."""
        return BatchQueue(
            redis_client=mock_redis,
            job_ttl_hours=24,
        )

    def test_key_generation(self, queue):
        """Test Redis key generation."""
        assert queue._queue_key("tenant-1") == "batch:queue:tenant-1"
        assert queue._job_key("job-123") == "batch:job:job-123"
        assert queue._results_key("job-123") == "batch:results:job-123"
        assert queue._channel_key("tenant-1") == "batch:channel:tenant-1"

    @pytest.mark.asyncio
    async def test_enqueue(self, queue, mock_redis):
        """Test enqueueing a batch job."""
        items = [
            {"subject": "Test 1", "body": "Body 1"},
            {"subject": "Test 2", "body": "Body 2"},
        ]

        job = await queue.enqueue(
            tenant_id="tenant-1",
            items=items,
            job_type="email_batch",
            priority=JobPriority.NORMAL,
        )

        assert job.job_id is not None
        assert job.tenant_id == "tenant-1"
        assert job.total_items == 2
        assert job.status == JobStatus.QUEUED
        assert len(job.items) == 2

        # Verify Redis calls
        mock_redis.set.assert_called_once()
        mock_redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_enqueue_with_webhook(self, queue, mock_redis):
        """Test enqueueing with webhook callback."""
        job = await queue.enqueue(
            tenant_id="tenant-1",
            items=[{"data": "test"}],
            webhook_url="https://example.com/callback",
            webhook_secret="secret123",
        )

        assert job.webhook_url == "https://example.com/callback"
        assert job.webhook_secret == "secret123"

    @pytest.mark.asyncio
    async def test_dequeue(self, queue, mock_redis):
        """Test dequeueing a job."""
        job_data = BatchJob(
            job_id="job-123",
            tenant_id="tenant-1",
            total_items=5,
        )

        mock_redis.zpopmin.return_value = [(b"job-123", 5.0)]
        mock_redis.get.return_value = job_data.model_dump_json()

        dequeued = await queue.dequeue("tenant-1")

        assert dequeued is not None
        assert dequeued.job_id == "job-123"
        assert dequeued.status == JobStatus.PROCESSING
        assert dequeued.started_at is not None

    @pytest.mark.asyncio
    async def test_dequeue_empty(self, queue, mock_redis):
        """Test dequeueing from empty queue."""
        mock_redis.zpopmin.return_value = []

        dequeued = await queue.dequeue("tenant-1")

        assert dequeued is None

    @pytest.mark.asyncio
    async def test_get_job(self, queue, mock_redis):
        """Test getting a job by ID."""
        job_data = BatchJob(
            job_id="job-456",
            tenant_id="tenant-1",
            total_items=10,
        )
        mock_redis.get.return_value = job_data.model_dump_json()

        job = await queue.get_job("job-456")

        assert job is not None
        assert job.job_id == "job-456"

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, queue, mock_redis):
        """Test getting non-existent job."""
        mock_redis.get.return_value = None

        job = await queue.get_job("nonexistent")

        assert job is None

    @pytest.mark.asyncio
    async def test_update_job_status(self, queue, mock_redis):
        """Test updating job status."""
        job_data = BatchJob(
            job_id="job-789",
            tenant_id="tenant-1",
            status=JobStatus.PROCESSING,
        )
        mock_redis.get.return_value = job_data.model_dump_json()

        updated = await queue.update_job_status(
            "job-789",
            JobStatus.COMPLETED,
        )

        assert updated.status == JobStatus.COMPLETED
        assert updated.completed_at is not None
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_job_progress(self, queue, mock_redis):
        """Test updating job progress."""
        job_data = BatchJob(
            job_id="job-progress",
            tenant_id="tenant-1",
            total_items=100,
            status=JobStatus.PROCESSING,
        )
        mock_redis.get.return_value = job_data.model_dump_json()

        updated = await queue.update_job_progress(
            "job-progress",
            processed=50,
            failed=2,
        )

        assert updated.processed_items == 50
        assert updated.failed_items == 2
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_result(self, queue, mock_redis):
        """Test adding a result for an item."""
        await queue.add_result(
            job_id="job-results",
            item_id="item-1",
            result={"intent": "ORDER_STATUS.WISMO"},
            success=True,
        )

        mock_redis.rpush.assert_called_once()
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_result_with_error(self, queue, mock_redis):
        """Test adding a failed result."""
        await queue.add_result(
            job_id="job-results",
            item_id="item-2",
            result={},
            success=False,
            error="Processing failed",
        )

        call_args = mock_redis.rpush.call_args
        result_json = call_args[0][1]
        result = json.loads(result_json)

        assert result["success"] is False
        assert result["error"] == "Processing failed"

    @pytest.mark.asyncio
    async def test_get_results(self, queue, mock_redis):
        """Test getting job results."""
        mock_redis.lrange.return_value = [
            json.dumps({"item_id": "1", "success": True, "result": {}}),
            json.dumps({"item_id": "2", "success": False, "error": "Failed"}),
        ]

        results = await queue.get_results("job-123")

        assert len(results) == 2
        assert results[0]["item_id"] == "1"
        assert results[1]["success"] is False

    @pytest.mark.asyncio
    async def test_cancel_job(self, queue, mock_redis):
        """Test canceling a queued job."""
        job_data = BatchJob(
            job_id="job-cancel",
            tenant_id="tenant-1",
            status=JobStatus.QUEUED,
        )
        mock_redis.get.return_value = job_data.model_dump_json()

        result = await queue.cancel_job("job-cancel")

        assert result is True
        mock_redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_processing_job_fails(self, queue, mock_redis):
        """Test that canceling a processing job fails."""
        job_data = BatchJob(
            job_id="job-processing",
            tenant_id="tenant-1",
            status=JobStatus.PROCESSING,
        )
        mock_redis.get.return_value = job_data.model_dump_json()

        result = await queue.cancel_job("job-processing")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_queue_length(self, queue, mock_redis):
        """Test getting queue length."""
        mock_redis.zcard.return_value = 5

        length = await queue.get_queue_length("tenant-1")

        assert length == 5


class TestJobPriority:
    """Tests for job priority."""

    def test_priority_ordering(self):
        """Test that high priority has lower numeric value."""
        assert JobPriority.HIGH.value < JobPriority.NORMAL.value
        assert JobPriority.NORMAL.value < JobPriority.LOW.value
