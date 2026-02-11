"""API routes for the customer service agent."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from intent_engine.agents import (
    CustomerMessage,
    CustomerServiceAgent,
    LifecycleRouter,
    get_catalog_provider_from_settings,
)
from intent_engine.api.middleware import verify_api_key
from intent_engine.config import get_settings

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/v1/agent", tags=["Agent"])
router = api_router  # Alias for server.include_router(agent_router)

# Global agent instance
_agent: CustomerServiceAgent | None = None
_router: LifecycleRouter | None = None


async def get_agent() -> CustomerServiceAgent:
    """Get or initialize the customer service agent."""
    global _agent
    if _agent is None:
        settings = get_settings()
        _agent = CustomerServiceAgent(settings=settings)
        await _agent.initialize()
    return _agent


async def get_router(
    agent: CustomerServiceAgent = Depends(get_agent),
) -> LifecycleRouter:
    """Get the lifecycle router (routes to pre-purchase or post-purchase agent)."""
    global _router
    if _router is None:
        _router = LifecycleRouter(
            intent_engine=agent.intent_engine,
            customer_service_agent=agent,
            catalog_provider=get_catalog_provider_from_settings(),
        )
    return _router


def set_agent(agent: CustomerServiceAgent | None) -> None:
    """Set the agent instance (for testing)."""
    global _agent
    _agent = agent


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message_id: str = Field(description="Unique message identifier")
    conversation_id: str | None = Field(default=None, description="Conversation thread ID")
    customer_email: str | None = Field(default=None, description="Customer email for lookup")
    customer_id: str | None = Field(default=None, description="Customer ID if known")
    text: str = Field(description="The customer's message")
    channel: str = Field(default="chat", description="Input channel")
    platform: str | None = Field(default=None, description="Platform (shopify, adobe_commerce)")
    order_ids: list[str] = Field(default_factory=list, description="Known order IDs")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": "msg-123",
                    "customer_email": "john@example.com",
                    "text": "Where is my order #12345?",
                    "platform": "shopify",
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    message_id: str
    conversation_id: str | None = None
    response_text: str
    intents: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    requires_human: bool = False
    confidence: float = 0.0
    processing_time_ms: int = 0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": "msg-123",
                    "response_text": "Your order #12345 is currently in transit.",
                    "intents": [
                        {"category": "ORDER_STATUS", "intent": "WISMO", "confidence": 0.92}
                    ],
                    "confidence": 0.92,
                    "processing_time_ms": 150,
                }
            ]
        }
    }


@api_router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Process customer message",
    description="Send a customer message and receive an AI-generated response with actions. Routes to pre-purchase (product/discovery) or post-purchase agent by intent.",
)
async def chat(
    body: ChatRequest,
    _api_key: str = Depends(verify_api_key),
    router: LifecycleRouter = Depends(get_router),
) -> ChatResponse:
    """
    Process a customer message through the lifecycle router.

    The router classifies intent and:
    - For product/discovery intents: uses the pre-purchase agent (catalog).
    - For order/return/complaint/etc.: uses the customer service agent (post-purchase).
    """
    message = CustomerMessage(
        message_id=body.message_id,
        conversation_id=body.conversation_id,
        customer_email=body.customer_email,
        customer_id=body.customer_id,
        text=body.text,
        channel=body.channel,
        platform=body.platform,
        order_ids=body.order_ids,
        metadata=body.metadata,
    )

    try:
        result = await router.process_message(message)

        return ChatResponse(
            message_id=result.message_id,
            conversation_id=result.conversation_id,
            response_text=result.response_text,
            intents=result.intents,
            actions=[action.model_dump() for action in result.actions],
            requires_human=result.requires_human,
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
        )

    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}",
        )


@api_router.get(
    "/health",
    summary="Agent health check",
    description="Check if the agent is initialized and ready.",
)
async def agent_health() -> dict[str, Any]:
    """Check agent health and available connectors."""
    global _agent

    if _agent is None:
        return {
            "status": "not_initialized",
            "connectors": [],
        }

    connectors = list(_agent._connectors.keys()) if _agent._connectors else []

    return {
        "status": "healthy" if _agent._initialized else "initializing",
        "connectors": connectors,
        "llm_available": _agent._llm_client is not None,
    }


@api_router.delete(
    "/conversations/{conversation_id}",
    summary="Clear conversation",
    description="Clear the context for a conversation.",
)
async def clear_conversation(
    conversation_id: str,
    _api_key: str = Depends(verify_api_key),
    agent: CustomerServiceAgent = Depends(get_agent),
) -> dict[str, str]:
    """Clear conversation context."""
    if conversation_id in agent._conversations:
        del agent._conversations[conversation_id]
        return {"status": "cleared", "conversation_id": conversation_id}
    return {"status": "not_found", "conversation_id": conversation_id}
