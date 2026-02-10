"""A2A Task Handler - Executes tasks submitted by other agents."""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from intent_engine.engine import IntentEngine
from intent_engine.models.request import InputChannel, IntentRequest
from intent_engine.storage.intent_catalog import IntentCatalogStore

logger = logging.getLogger(__name__)


class TaskStatus(StrEnum):
    """A2A task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2ATask(BaseModel):
    """An A2A task submitted for execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action: str = Field(description="The action to execute")
    input: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


class TaskSubmission(BaseModel):
    """Request body for task submission."""

    action: str = Field(description="The action to execute")
    input: dict[str, Any] = Field(default_factory=dict)
    async_mode: bool = Field(
        default=False,
        description="If true, returns immediately with task ID for polling",
    )


class A2ATaskHandler:
    """
    Handles A2A task execution.

    Manages task lifecycle: submission, execution, status tracking, and cancellation.
    """

    def __init__(self, engine: IntentEngine) -> None:
        """
        Initialize the task handler.

        Args:
            engine: The intent engine instance for task execution.
        """
        self.engine = engine
        self._tasks: dict[str, A2ATask] = {}
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    async def submit_task(self, submission: TaskSubmission) -> A2ATask:
        """
        Submit a new task for execution.

        Args:
            submission: Task submission request.

        Returns:
            The created task (with result if sync, or pending if async).
        """
        task = A2ATask(
            action=submission.action,
            input=submission.input,
        )
        self._tasks[task.id] = task

        if submission.async_mode:
            # Start task in background
            asyncio_task = asyncio.create_task(self._execute_task(task.id))
            self._running_tasks[task.id] = asyncio_task
            return task
        else:
            # Execute synchronously
            await self._execute_task(task.id)
            return self._tasks[task.id]

    async def _execute_task(self, task_id: str) -> None:
        """Execute a task and update its status."""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.RUNNING

        try:
            if task.action == "resolve_intent":
                result = await self._execute_resolve_intent(task.input)
            elif task.action == "classify_intent_fast":
                result = await self._execute_classify_fast(task.input)
            elif task.action == "list_intent_taxonomy":
                result = await self._execute_list_taxonomy(task.input)
            else:
                raise ValueError(f"Unknown action: {task.action}")

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            raise

        except Exception as e:
            logger.exception(f"Task {task_id} failed")
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()

        finally:
            # Clean up running task reference
            self._running_tasks.pop(task_id, None)

    async def _execute_resolve_intent(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute resolve_intent action."""
        raw_text = input_data.get("raw_text", "")
        if not raw_text:
            raise ValueError("raw_text is required")

        request = IntentRequest(
            request_id=f"a2a-{uuid.uuid4().hex[:8]}",
            tenant_id="a2a-agent",
            channel=InputChannel.CHAT,
            raw_text=raw_text,
            customer_tier=input_data.get("customer_tier"),
            order_ids=input_data.get("order_ids", []),
            previous_intents=input_data.get("previous_intents", []),
        )

        result = await self.engine.resolve(request)

        return {
            "request_id": result.request_id,
            "resolved_intents": [
                {
                    "category": intent.category,
                    "intent": intent.intent,
                    "confidence": intent.confidence,
                    "confidence_tier": intent.confidence_tier.value,
                    "evidence": intent.evidence,
                }
                for intent in result.resolved_intents
            ],
            "is_compound": result.is_compound,
            "entities": [
                {
                    "entity_type": entity.entity_type,
                    "value": entity.value,
                    "confidence": entity.confidence,
                }
                for entity in result.entities
            ],
            "confidence_summary": result.confidence_summary,
            "requires_human": result.requires_human,
            "human_handoff_reason": result.human_handoff_reason,
            "path_taken": result.path_taken,
            "processing_time_ms": result.processing_time_ms,
        }

    async def _execute_classify_fast(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute classify_intent_fast action."""
        raw_text = input_data.get("raw_text", "")
        if not raw_text:
            raise ValueError("raw_text is required")

        # Generate embedding and match
        embedding = self.engine.components.embedding_extractor.embed(raw_text)
        match_result = await self.engine.components.intent_matcher.match(
            text=raw_text,
            embedding=embedding,
        )

        result: dict[str, Any] = {
            "decision": match_result.decision.value,
            "top_matches": [
                {
                    "intent_code": match.intent_code,
                    "similarity": round(match.similarity, 4),
                    "matched_example": match.matched_example[:100],
                }
                for match in match_result.top_matches[:5]
            ],
        }

        if match_result.resolved_intent:
            result["resolved_intent"] = {
                "category": match_result.resolved_intent.category,
                "intent": match_result.resolved_intent.intent,
                "confidence": match_result.resolved_intent.confidence,
            }

        return result

    async def _execute_list_taxonomy(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute list_intent_taxonomy action."""
        catalog_store = IntentCatalogStore(
            vector_store=None,  # type: ignore - not needed for static data
            embedding_extractor=None,
        )
        intents = catalog_store.get_core_intents()

        return {
            "intent_count": len(intents),
            "intents": intents,
            "categories": list(set(i["category"] for i in intents)),
        }

    def get_task(self, task_id: str) -> A2ATask | None:
        """
        Get a task by ID.

        Args:
            task_id: The task ID.

        Returns:
            The task, or None if not found.
        """
        return self._tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            True if task was cancelled, False if not found or not running.
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            return False

        # Cancel the asyncio task if running
        asyncio_task = self._running_tasks.get(task_id)
        if asyncio_task and not asyncio_task.done():
            asyncio_task.cancel()
            try:
                await asyncio_task
            except asyncio.CancelledError:
                pass

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        return True

    def cleanup_old_tasks(self, max_age_seconds: int = 3600) -> int:
        """
        Remove completed tasks older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age of tasks to keep.

        Returns:
            Number of tasks removed.
        """
        now = datetime.utcnow()
        to_remove = []

        for task_id, task in self._tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self._tasks[task_id]

        return len(to_remove)
