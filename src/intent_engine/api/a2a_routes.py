"""A2A protocol routes for FastAPI."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from intent_engine.a2a.agent_card import get_agent_card
from intent_engine.a2a.handler import A2ATask, A2ATaskHandler, TaskSubmission
from intent_engine.api.routes import get_engine
from intent_engine.engine import IntentEngine

a2a_router = APIRouter(tags=["A2A Protocol"])

# Task handler instance (initialized on first request)
_task_handler: A2ATaskHandler | None = None


def get_task_handler(engine: IntentEngine = Depends(get_engine)) -> A2ATaskHandler:
    """Get or create the A2A task handler."""
    global _task_handler
    if _task_handler is None:
        _task_handler = A2ATaskHandler(engine)
    return _task_handler


@a2a_router.get(
    "/.well-known/agent.json",
    summary="Get agent card",
    description="Returns the A2A agent card describing this agent's capabilities.",
    response_class=JSONResponse,
)
async def get_agent_card_endpoint() -> dict:
    """
    Return the agent card for A2A discovery.

    This endpoint follows the A2A protocol specification for agent advertisement.
    Other agents can fetch this to understand what this agent can do.
    """
    card = get_agent_card(base_url=None)  # Base URL will be inferred by caller
    return card.model_dump(exclude_none=True)


@a2a_router.post(
    "/a2a/tasks",
    response_model=A2ATask,
    summary="Submit task",
    description="Submit a task for execution. Use async_mode=true for background execution.",
)
async def submit_task(
    submission: TaskSubmission,
    handler: A2ATaskHandler = Depends(get_task_handler),
) -> A2ATask:
    """
    Submit a task for execution.

    Supported actions:
    - resolve_intent: Full intent resolution with reasoning
    - classify_intent_fast: Quick embedding-based classification
    - list_intent_taxonomy: Get supported intent categories

    Set async_mode=true to run in background and poll for results.
    """
    # Validate action
    valid_actions = ["resolve_intent", "classify_intent_fast", "list_intent_taxonomy"]
    if submission.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {submission.action}. Valid actions: {valid_actions}",
        )

    task = await handler.submit_task(submission)
    return task


@a2a_router.get(
    "/a2a/tasks/{task_id}",
    response_model=A2ATask,
    summary="Get task status",
    description="Get the status and result of a task.",
)
async def get_task_status(
    task_id: str,
    handler: A2ATaskHandler = Depends(get_task_handler),
) -> A2ATask:
    """
    Get the status and result of a submitted task.

    Returns the task with its current status, and result if completed.
    """
    task = handler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return task


@a2a_router.post(
    "/a2a/tasks/{task_id}/cancel",
    response_model=dict,
    summary="Cancel task",
    description="Cancel a pending or running task.",
)
async def cancel_task(
    task_id: str,
    handler: A2ATaskHandler = Depends(get_task_handler),
) -> dict:
    """
    Cancel a running or pending task.

    Returns success status.
    """
    task = handler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    cancelled = await handler.cancel_task(task_id)

    if cancelled:
        return {"status": "cancelled", "task_id": task_id}
    else:
        return {
            "status": "not_cancelled",
            "task_id": task_id,
            "reason": f"Task is in status: {task.status.value}",
        }
