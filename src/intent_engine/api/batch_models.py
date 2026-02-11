"""Batch API request and response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BatchEmailItem(BaseModel):
    """Single email item for batch processing."""

    # Option 1: Raw MIME email
    raw_email: str | None = None

    # Option 2: Structured email data
    subject: str | None = None
    body: str | None = None
    from_email: str | None = None
    from_name: str | None = None
    to_email: str | None = None
    message_id: str | None = None

    # Optional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateBatchJobRequest(BaseModel):
    """Request to create a batch job."""

    items: list[BatchEmailItem] = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="List of emails to process",
    )
    priority: str = Field(
        default="normal",
        description="Job priority: high, normal, low",
    )
    webhook_url: str | None = Field(
        default=None,
        description="URL to call when job completes",
    )
    webhook_secret: str | None = Field(
        default=None,
        description="Secret for webhook HMAC signature",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "subject": "Order issue",
                            "body": "Where is my order #1234?",
                            "from_email": "customer@example.com",
                        },
                        {
                            "subject": "Return request",
                            "body": "I want to return item ABC",
                            "from_email": "other@example.com",
                        },
                    ],
                    "priority": "normal",
                }
            ]
        }
    }


class BatchJobResponse(BaseModel):
    """Response for batch job operations."""

    job_id: str
    tenant_id: str
    status: str
    job_type: str = "email_batch"
    total_items: int
    processed_items: int = 0
    failed_items: int = 0
    progress: float = 0.0
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "550e8400-e29b-41d4-a716-446655440000",
                    "tenant_id": "tenant-001",
                    "status": "queued",
                    "job_type": "email_batch",
                    "total_items": 100,
                    "progress": 0.0,
                    "created_at": "2024-01-15T10:30:00Z",
                }
            ]
        }
    }


class BatchResultItem(BaseModel):
    """Result for a single batch item."""

    item_id: str
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    processed_at: datetime | None = None


class BatchJobResultsResponse(BaseModel):
    """Response with full batch results."""

    job_id: str
    status: str
    total_items: int
    processed_items: int
    failed_items: int
    duration_seconds: float | None = None
    results: list[BatchResultItem] = Field(default_factory=list)


class BatchJobListResponse(BaseModel):
    """Response for listing batch jobs."""

    jobs: list[BatchJobResponse]
    total: int
    page: int = 1
    page_size: int = 20
