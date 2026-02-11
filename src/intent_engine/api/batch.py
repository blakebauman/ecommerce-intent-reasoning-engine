"""Batch processing API endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from intent_engine.api.batch_models import (
    BatchJobListResponse,
    BatchJobResponse,
    BatchJobResultsResponse,
    BatchResultItem,
    CreateBatchJobRequest,
)
from intent_engine.batch.queue import BatchQueue, JobPriority, JobStatus
from intent_engine.tenancy.context import get_current_tenant
from intent_engine.tenancy.models import TenantConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/batch", tags=["batch"])

# Module-level storage for dependency overrides (used in testing)
_queue_override: BatchQueue | None = None
_tenant_override: TenantConfig | None = None


def set_queue_override(queue: BatchQueue | None) -> None:
    """Set queue override for testing."""
    global _queue_override
    _queue_override = queue


def set_tenant_override(tenant: TenantConfig | None) -> None:
    """Set tenant override for testing."""
    global _tenant_override
    _tenant_override = tenant


def get_queue() -> BatchQueue:
    """
    Get the batch queue instance.

    This is a placeholder - should be set up during app initialization.
    """
    global _queue_override
    if _queue_override is not None:
        return _queue_override

    # Lazy import to avoid circular dependencies
    try:
        from intent_engine.api.server import get_batch_queue

        queue = get_batch_queue()
    except ImportError:
        queue = None

    if queue is None:
        raise HTTPException(
            status_code=503,
            detail="Batch processing not available",
        )
    return queue


def get_tenant() -> TenantConfig:
    """Get the current tenant from context."""
    global _tenant_override
    if _tenant_override is not None:
        return _tenant_override

    tenant = get_current_tenant()
    if tenant is None:
        raise HTTPException(
            status_code=401,
            detail="Tenant context not available",
        )
    return tenant


@router.post(
    "/emails",
    response_model=BatchJobResponse,
    summary="Create email batch job",
    description="Submit a batch of emails for intent resolution.",
)
async def create_email_batch(
    request: CreateBatchJobRequest,
    queue: BatchQueue = Depends(get_queue),
    tenant: TenantConfig = Depends(get_tenant),
) -> BatchJobResponse:
    """
    Create a batch job to process multiple emails.

    The job will be queued and processed by background workers.
    Use the job_id to poll for status or subscribe via WebSocket.
    """
    # Check if batch processing is enabled for tenant
    if not tenant.batch_processing_enabled:
        raise HTTPException(
            status_code=403,
            detail="Batch processing not enabled for this tenant",
        )

    # Check batch size limit
    max_size = tenant.get_max_batch_size()
    if len(request.items) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds limit: {len(request.items)} > {max_size}",
        )

    # Map priority string to enum
    priority_map = {
        "high": JobPriority.HIGH,
        "normal": JobPriority.NORMAL,
        "low": JobPriority.LOW,
    }
    priority = priority_map.get(request.priority.lower(), JobPriority.NORMAL)

    # Convert items to dicts for queue
    items = [item.model_dump(exclude_none=True) for item in request.items]

    # Enqueue the job
    job = await queue.enqueue(
        tenant_id=tenant.tenant_id,
        items=items,
        job_type="email_batch",
        priority=priority,
        webhook_url=request.webhook_url,
        webhook_secret=request.webhook_secret,
        metadata=request.metadata,
    )

    logger.info(f"Created batch job {job.job_id} for tenant {tenant.tenant_id}")

    return BatchJobResponse(
        job_id=job.job_id,
        tenant_id=job.tenant_id,
        status=job.status.value,
        job_type=job.job_type,
        total_items=job.total_items,
        processed_items=job.processed_items,
        failed_items=job.failed_items,
        progress=job.progress(),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=BatchJobResponse,
    summary="Get job status",
    description="Get the current status of a batch job.",
)
async def get_job_status(
    job_id: str,
    queue: BatchQueue = Depends(get_queue),
    tenant: TenantConfig = Depends(get_tenant),
) -> BatchJobResponse:
    """Get the status of a batch job."""
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify tenant owns this job
    if job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    return BatchJobResponse(
        job_id=job.job_id,
        tenant_id=job.tenant_id,
        status=job.status.value,
        job_type=job.job_type,
        total_items=job.total_items,
        processed_items=job.processed_items,
        failed_items=job.failed_items,
        progress=job.progress(),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@router.get(
    "/jobs/{job_id}/results",
    response_model=BatchJobResultsResponse,
    summary="Get job results",
    description="Get the results of a completed batch job.",
)
async def get_job_results(
    job_id: str,
    queue: BatchQueue = Depends(get_queue),
    tenant: TenantConfig = Depends(get_tenant),
) -> BatchJobResultsResponse:
    """Get the results of a batch job."""
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify tenant owns this job
    if job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check if job is complete
    if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )

    # Get results
    results_data = await queue.get_results(job_id)
    results = [
        BatchResultItem(
            item_id=r["item_id"],
            success=r["success"],
            result=r.get("result"),
            error=r.get("error"),
            processed_at=r.get("processed_at"),
        )
        for r in results_data
    ]

    return BatchJobResultsResponse(
        job_id=job.job_id,
        status=job.status.value,
        total_items=job.total_items,
        processed_items=job.processed_items,
        failed_items=job.failed_items,
        duration_seconds=job.duration_seconds(),
        results=results,
    )


@router.delete(
    "/jobs/{job_id}",
    summary="Cancel job",
    description="Cancel a queued batch job.",
)
async def cancel_job(
    job_id: str,
    queue: BatchQueue = Depends(get_queue),
    tenant: TenantConfig = Depends(get_tenant),
) -> dict[str, Any]:
    """Cancel a queued batch job."""
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify tenant owns this job
    if job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Try to cancel
    cancelled = await queue.cancel_job(job_id)

    if not cancelled:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}",
        )

    return {"job_id": job_id, "status": "cancelled"}


@router.get(
    "/jobs",
    response_model=BatchJobListResponse,
    summary="List jobs",
    description="List batch jobs for the current tenant.",
)
async def list_jobs(
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
    queue: BatchQueue = Depends(get_queue),
    tenant: TenantConfig = Depends(get_tenant),
) -> BatchJobListResponse:
    """List batch jobs for the current tenant."""
    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}",
            )

    # Get jobs
    jobs = await queue.list_jobs(
        tenant_id=tenant.tenant_id,
        status=status_filter,
        limit=page_size,
    )

    # Convert to response models
    job_responses = [
        BatchJobResponse(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=job.status.value,
            job_type=job.job_type,
            total_items=job.total_items,
            processed_items=job.processed_items,
            failed_items=job.failed_items,
            progress=job.progress(),
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error=job.error,
        )
        for job in jobs
    ]

    return BatchJobListResponse(
        jobs=job_responses,
        total=len(job_responses),
        page=page,
        page_size=page_size,
    )
