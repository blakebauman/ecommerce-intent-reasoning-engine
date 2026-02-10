"""Background worker for batch job processing."""

import asyncio
import hashlib
import hmac
import logging
import time
from typing import Any, Callable

import httpx

from intent_engine.batch.queue import BatchJob, BatchQueue, JobStatus
from intent_engine.observability.metrics import record_batch_job
from intent_engine.tenancy.context import tenant_context
from intent_engine.tenancy.models import TenantConfig, TenantTier

logger = logging.getLogger(__name__)


class BatchWorker:
    """
    Background worker for processing batch jobs.

    Features:
    - Configurable concurrency
    - Graceful shutdown
    - Webhook callbacks
    - Per-tenant rate limiting
    - Metrics recording
    """

    def __init__(
        self,
        queue: BatchQueue,
        process_item: Callable[[dict[str, Any], str], Any],
        tenant_lookup: Callable[[str], TenantConfig | None] | None = None,
        concurrency: int = 5,
        poll_interval: float = 1.0,
    ) -> None:
        """
        Initialize the batch worker.

        Args:
            queue: BatchQueue instance.
            process_item: Async function to process a single item.
                         Takes (item_payload, tenant_id) and returns result.
            tenant_lookup: Optional function to get tenant config by ID.
            concurrency: Number of concurrent item processors.
            poll_interval: Seconds between queue polls.
        """
        self.queue = queue
        self.process_item = process_item
        self.tenant_lookup = tenant_lookup
        self.concurrency = concurrency
        self.poll_interval = poll_interval

        self._running = False
        self._tasks: set[asyncio.Task] = set()
        self._http_client: httpx.AsyncClient | None = None

    async def start(self, tenant_ids: list[str] | None = None) -> None:
        """
        Start the worker.

        Args:
            tenant_ids: List of tenant IDs to process. If None, processes all.
        """
        if self._running:
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=30.0)

        logger.info(
            f"Starting batch worker: concurrency={self.concurrency}, "
            f"poll_interval={self.poll_interval}s"
        )

        # Start processing loop
        if tenant_ids:
            # Process specific tenants
            for tenant_id in tenant_ids:
                task = asyncio.create_task(self._worker_loop(tenant_id))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        else:
            # Single worker for all tenants (simpler for now)
            # In production, would discover tenants from queue keys
            task = asyncio.create_task(self._generic_worker_loop())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        if not self._running:
            return

        logger.info("Stopping batch worker...")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("Batch worker stopped")

    async def _worker_loop(self, tenant_id: str) -> None:
        """Worker loop for a specific tenant."""
        logger.info(f"Worker started for tenant: {tenant_id}")

        while self._running:
            try:
                job = await self.queue.dequeue(tenant_id)

                if job:
                    await self._process_job(job)
                else:
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in worker loop for {tenant_id}: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _generic_worker_loop(self) -> None:
        """Worker loop that checks multiple tenant queues."""
        # For simplicity, maintain a list of known tenants
        # In production, this would be more sophisticated
        known_tenants: set[str] = set()

        while self._running:
            try:
                # Try to find work from any known tenant
                processed = False

                for tenant_id in list(known_tenants):
                    job = await self.queue.dequeue(tenant_id)
                    if job:
                        await self._process_job(job)
                        processed = True
                        break

                if not processed:
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in generic worker loop: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _process_job(self, job: BatchJob) -> None:
        """
        Process a single batch job.

        Args:
            job: The job to process.
        """
        start_time = time.time()
        logger.info(f"Processing job {job.job_id}: {job.total_items} items")

        # Get tenant config if available
        tenant = None
        if self.tenant_lookup:
            tenant = self.tenant_lookup(job.tenant_id)
            if hasattr(tenant, "__await__"):
                tenant = await tenant

        # Use default tenant if not found
        if tenant is None:
            tenant = TenantConfig(
                tenant_id=job.tenant_id,
                name="Unknown",
                tier=TenantTier.STARTER,
                api_key="",
            )

        try:
            # Process items with concurrency limit
            semaphore = asyncio.Semaphore(self.concurrency)
            processed = 0
            failed = 0

            async def process_with_semaphore(item):
                nonlocal processed, failed
                async with semaphore:
                    try:
                        with tenant_context(tenant):
                            result = await self.process_item(item.payload, job.tenant_id)

                        await self.queue.add_result(
                            job.job_id,
                            item.item_id,
                            result if isinstance(result, dict) else {"result": result},
                            success=True,
                        )
                        processed += 1

                    except Exception as e:
                        logger.warning(f"Failed to process item {item.item_id}: {e}")
                        await self.queue.add_result(
                            job.job_id,
                            item.item_id,
                            {},
                            success=False,
                            error=str(e),
                        )
                        failed += 1

                    # Update progress periodically
                    if (processed + failed) % 10 == 0:
                        await self.queue.update_job_progress(
                            job.job_id,
                            processed=processed + failed,
                            failed=failed,
                        )

            # Process all items concurrently
            await asyncio.gather(*[
                process_with_semaphore(item)
                for item in job.items
            ])

            # Update final status
            status = JobStatus.COMPLETED if failed == 0 else JobStatus.FAILED
            job = await self.queue.update_job_status(
                job.job_id,
                status,
                error=f"{failed} items failed" if failed > 0 else None,
            )

            duration = time.time() - start_time
            logger.info(
                f"Completed job {job.job_id}: "
                f"{processed} processed, {failed} failed, "
                f"{duration:.2f}s"
            )

            # Record metrics
            record_batch_job(
                tenant_id=job.tenant_id,
                item_count=job.total_items,
                duration_seconds=duration,
                status=status.value,
            )

            # Send webhook callback
            if job.webhook_url:
                await self._send_webhook(job)

        except Exception as e:
            logger.exception(f"Error processing job {job.job_id}: {e}")
            await self.queue.update_job_status(
                job.job_id,
                JobStatus.FAILED,
                error=str(e),
            )

            # Record failure metric
            record_batch_job(
                tenant_id=job.tenant_id,
                item_count=job.total_items,
                duration_seconds=time.time() - start_time,
                status="failed",
            )

    async def _send_webhook(self, job: BatchJob) -> None:
        """
        Send webhook callback for completed job.

        Args:
            job: The completed job.
        """
        if not job.webhook_url or not self._http_client:
            return

        try:
            # Build payload
            results = await self.queue.get_results(job.job_id)
            payload = {
                "job_id": job.job_id,
                "tenant_id": job.tenant_id,
                "status": job.status.value,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "failed_items": job.failed_items,
                "duration_seconds": job.duration_seconds(),
                "results": results,
                "error": job.error,
            }

            import json
            body = json.dumps(payload)

            headers = {"Content-Type": "application/json"}

            # Add HMAC signature if secret provided
            if job.webhook_secret:
                signature = hmac.new(
                    job.webhook_secret.encode(),
                    body.encode(),
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Webhook-Signature"] = signature

            response = await self._http_client.post(
                job.webhook_url,
                content=body,
                headers=headers,
            )
            response.raise_for_status()

            logger.info(f"Webhook sent for job {job.job_id}")

        except Exception as e:
            logger.warning(f"Failed to send webhook for job {job.job_id}: {e}")


async def create_email_processor(engine_getter: Callable):
    """
    Create a processor function for email batch jobs.

    Args:
        engine_getter: Function to get the IntentEngine instance.

    Returns:
        Async function to process email items.
    """
    from intent_engine.ingestion.email import EmailAdapter

    adapter = EmailAdapter()

    async def process_email(payload: dict, tenant_id: str) -> dict:
        """Process a single email item."""
        engine = engine_getter()
        if engine is None:
            raise RuntimeError("Engine not initialized")

        # Parse email into IntentRequest
        request = adapter.parse(payload)

        # Override tenant ID
        request.tenant_id = tenant_id

        # Resolve intent
        result = await engine.resolve(request)

        return result.model_dump()

    return process_email
