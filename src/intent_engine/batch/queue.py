"""Redis-based batch job queue."""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, cast

import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Batch job status."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels (lower = higher priority)."""

    HIGH = 1
    NORMAL = 5
    LOW = 10


class BatchJobItem(BaseModel):
    """Single item in a batch job."""

    item_id: str
    payload: dict[str, Any]
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None


class BatchJob(BaseModel):
    """Batch job model."""

    job_id: str
    tenant_id: str
    status: JobStatus = JobStatus.QUEUED
    priority: int = JobPriority.NORMAL.value
    job_type: str = "email_batch"  # email_batch, bulk_resolve, etc.

    # Items
    items: list[BatchJobItem] = Field(default_factory=list)
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Webhook callback
    webhook_url: str | None = None
    webhook_secret: str | None = None

    # Results
    results: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def progress(self) -> float:
        """Calculate job progress (0.0 to 1.0)."""
        if self.total_items == 0:
            return 0.0
        return self.processed_items / self.total_items

    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class BatchQueue:
    """
    Redis-based priority queue for batch jobs.

    Uses:
    - Sorted set for priority queue
    - Hash for job data
    - Pub/sub for job notifications

    Keys:
    - batch:queue:{tenant_id} - Sorted set of job IDs (score = priority + timestamp)
    - batch:job:{job_id} - Hash with job data
    - batch:results:{job_id} - List of results
    - batch:channel:{tenant_id} - Pub/sub channel for updates
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        job_ttl_hours: int = 24,
    ) -> None:
        """
        Initialize the batch queue.

        Args:
            redis_client: Async Redis client.
            job_ttl_hours: Hours to keep completed jobs.
        """
        self.redis = redis_client
        self.job_ttl_seconds = job_ttl_hours * 3600

    def _queue_key(self, tenant_id: str) -> str:
        """Get Redis key for tenant queue."""
        return f"batch:queue:{tenant_id}"

    def _job_key(self, job_id: str) -> str:
        """Get Redis key for job data."""
        return f"batch:job:{job_id}"

    def _results_key(self, job_id: str) -> str:
        """Get Redis key for job results."""
        return f"batch:results:{job_id}"

    def _channel_key(self, tenant_id: str) -> str:
        """Get Redis key for pub/sub channel."""
        return f"batch:channel:{tenant_id}"

    async def enqueue(
        self,
        tenant_id: str,
        items: list[dict[str, Any]],
        job_type: str = "email_batch",
        priority: JobPriority = JobPriority.NORMAL,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BatchJob:
        """
        Enqueue a new batch job.

        Args:
            tenant_id: Tenant ID.
            items: List of items to process.
            job_type: Type of batch job.
            priority: Job priority.
            webhook_url: Optional webhook URL for completion callback.
            webhook_secret: Secret for webhook HMAC.
            metadata: Optional metadata.

        Returns:
            The created BatchJob.
        """
        job_id = str(uuid.uuid4())

        # Create job items
        job_items = [
            BatchJobItem(
                item_id=str(uuid.uuid4()),
                payload=item,
            )
            for item in items
        ]

        job = BatchJob(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatus.QUEUED,
            priority=priority.value,
            job_type=job_type,
            items=job_items,
            total_items=len(items),
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            metadata=metadata or {},
        )

        # Store job data
        await self.redis.set(
            self._job_key(job_id),
            job.model_dump_json(),
            ex=self.job_ttl_seconds,
        )

        # Add to priority queue
        # Score = priority * 1e10 + timestamp (ensures FIFO within priority)
        score = priority.value * 1e10 + time.time()
        await self.redis.zadd(
            self._queue_key(tenant_id),
            {job_id: score},
        )

        logger.info(
            f"Enqueued batch job {job_id} for tenant {tenant_id}: "
            f"{len(items)} items, priority={priority.name}"
        )

        return job

    async def dequeue(self, tenant_id: str) -> BatchJob | None:
        """
        Dequeue the next job for processing.

        Uses atomic pop to prevent race conditions.

        Args:
            tenant_id: Tenant ID.

        Returns:
            The next job to process, or None if queue is empty.
        """
        # Pop the highest priority (lowest score) job
        result = await self.redis.zpopmin(self._queue_key(tenant_id), count=1)

        if not result:
            return None

        job_id = result[0][0]
        if isinstance(job_id, bytes):
            job_id = job_id.decode()

        # Get job data
        job = await self.get_job(job_id)

        if job:
            # Update status to processing
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await self._save_job(job)
            logger.info(f"Dequeued batch job {job_id}")

        return job

    async def get_job(self, job_id: str) -> BatchJob | None:
        """
        Get a job by ID.

        Args:
            job_id: The job ID.

        Returns:
            The BatchJob if found, None otherwise.
        """
        data = await self.redis.get(self._job_key(job_id))
        if not data:
            return None

        if isinstance(data, bytes):
            data = data.decode()

        return BatchJob.model_validate_json(data)

    async def _save_job(self, job: BatchJob) -> None:
        """Save job data to Redis."""
        await self.redis.set(
            self._job_key(job.job_id),
            job.model_dump_json(),
            ex=self.job_ttl_seconds,
        )

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None,
    ) -> BatchJob | None:
        """
        Update job status.

        Args:
            job_id: The job ID.
            status: New status.
            error: Optional error message.

        Returns:
            Updated BatchJob.
        """
        job = await self.get_job(job_id)
        if not job:
            return None

        job.status = status
        if error:
            job.error = error

        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = datetime.now(timezone.utc)

        await self._save_job(job)

        # Publish status update
        await self.redis.publish(
            self._channel_key(job.tenant_id),
            json.dumps(
                {
                    "job_id": job_id,
                    "status": status.value,
                    "progress": job.progress(),
                }
            ),
        )

        return job

    async def update_job_progress(
        self,
        job_id: str,
        processed: int,
        failed: int = 0,
    ) -> BatchJob | None:
        """
        Update job progress.

        Args:
            job_id: The job ID.
            processed: Number of items processed.
            failed: Number of items failed.

        Returns:
            Updated BatchJob.
        """
        job = await self.get_job(job_id)
        if not job:
            return None

        job.processed_items = processed
        job.failed_items = failed
        await self._save_job(job)

        # Publish progress update
        await self.redis.publish(
            self._channel_key(job.tenant_id),
            json.dumps(
                {
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.progress(),
                    "processed": processed,
                    "failed": failed,
                }
            ),
        )

        return job

    async def add_result(
        self,
        job_id: str,
        item_id: str,
        result: dict[str, Any],
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """
        Add a result for a batch item.

        Args:
            job_id: The job ID.
            item_id: The item ID.
            result: Result data.
            success: Whether processing succeeded.
            error: Optional error message.
        """
        result_entry = {
            "item_id": item_id,
            "success": success,
            "result": result,
            "error": error,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.redis.rpush(
            self._results_key(job_id),
            json.dumps(result_entry),
        )
        await self.redis.expire(self._results_key(job_id), self.job_ttl_seconds)

    async def get_results(self, job_id: str) -> list[dict[str, Any]]:
        """
        Get all results for a job.

        Args:
            job_id: The job ID.

        Returns:
            List of result entries.
        """
        results = await self.redis.lrange(self._results_key(job_id), 0, -1)
        return [cast(dict[str, Any], json.loads(r)) for r in results]

    async def get_queue_length(self, tenant_id: str) -> int:
        """Get number of jobs in queue for a tenant."""
        n = await self.redis.zcard(self._queue_key(tenant_id))
        return int(n)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Only works for queued jobs (not processing).

        Args:
            job_id: The job ID.

        Returns:
            True if cancelled, False otherwise.
        """
        job = await self.get_job(job_id)
        if not job:
            return False

        if job.status != JobStatus.QUEUED:
            return False

        # Remove from queue
        await self.redis.zrem(self._queue_key(job.tenant_id), job_id)

        # Update status
        await self.update_job_status(job_id, JobStatus.CANCELLED)

        logger.info(f"Cancelled batch job {job_id}")
        return True

    async def list_jobs(
        self,
        tenant_id: str,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[BatchJob]:
        """
        List jobs for a tenant.

        Args:
            tenant_id: Tenant ID.
            status: Optional status filter.
            limit: Maximum jobs to return.

        Returns:
            List of BatchJob objects.
        """
        # Get all job IDs in queue
        job_ids = await self.redis.zrange(self._queue_key(tenant_id), 0, limit - 1)

        jobs = []
        for job_id in job_ids:
            if isinstance(job_id, bytes):
                job_id = job_id.decode()
            job = await self.get_job(job_id)
            if job:
                if status is None or job.status == status:
                    jobs.append(job)

        return jobs
