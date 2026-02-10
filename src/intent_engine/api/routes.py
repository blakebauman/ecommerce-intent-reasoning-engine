"""API routes for the intent engine."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from intent_engine.api.middleware import verify_api_key
from intent_engine.engine import IntentEngine
from intent_engine.models.request import InputChannel, IntentRequest
from intent_engine.models.response import ReasoningResult

router = APIRouter()

# Global engine instance (initialized in server.py)
_engine: IntentEngine | None = None


def get_engine() -> IntentEngine:
    """Get the intent engine instance."""
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized",
        )
    return _engine


def set_engine(engine: IntentEngine) -> None:
    """Set the intent engine instance."""
    global _engine
    _engine = engine


class ResolveRequest(BaseModel):
    """Request body for the resolve endpoint."""

    request_id: str = Field(description="Unique request identifier")
    tenant_id: str = Field(description="Merchant/store identifier")
    channel: InputChannel = Field(default=InputChannel.CHAT)
    raw_text: str = Field(description="The customer message to classify")

    # Optional context
    conversation_id: str | None = None
    customer_id: str | None = None
    customer_tier: str | None = None
    order_ids: list[str] = Field(default_factory=list)
    previous_intents: list[str] = Field(default_factory=list)

    model_config = {"json_schema_extra": {"examples": [
        {
            "request_id": "req-12345",
            "tenant_id": "merchant-1",
            "channel": "chat",
            "raw_text": "Where is my order #ORD-98765?",
        }
    ]}}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    catalog_stats: dict[str, Any] | None = None


class CatalogStatsResponse(BaseModel):
    """Catalog statistics response."""

    total_examples: int
    num_intents: int
    by_intent: dict[str, int]
    by_category: dict[str, int]


@router.post(
    "/v1/intent/resolve",
    response_model=ReasoningResult,
    summary="Resolve intent from customer message",
    description="Classify and decompose customer intents from a text message.",
)
async def resolve_intent(
    body: ResolveRequest,
    request: Request,
    _api_key: str = Depends(verify_api_key),
    engine: IntentEngine = Depends(get_engine),
) -> ReasoningResult:
    """
    Resolve intent(s) from a customer message.

    This is the primary endpoint for intent classification. It:
    1. Extracts entities (order IDs, dates, etc.)
    2. Matches against the intent catalog
    3. Uses LLM reasoning for complex/compound intents
    4. Returns structured intent classification

    Fast path (~70-80% of requests): < 200ms
    Reasoning path (complex cases): 1-3 seconds
    """
    # Build IntentRequest from body
    intent_request = IntentRequest(
        request_id=body.request_id,
        tenant_id=body.tenant_id,
        channel=body.channel,
        raw_text=body.raw_text,
        conversation_id=body.conversation_id,
        customer_id=body.customer_id,
        customer_tier=body.customer_tier,
        order_ids=body.order_ids,
        previous_intents=body.previous_intents,
    )

    # Resolve intent
    result = await engine.resolve(intent_request)

    return result


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and ready to accept requests.",
)
async def health_check(
    engine: IntentEngine = Depends(get_engine),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status and catalog statistics.
    """
    from intent_engine.config import get_settings
    from intent_engine.storage.intent_catalog import IntentCatalogStore

    settings = get_settings()

    try:
        # Get catalog stats
        catalog_store = IntentCatalogStore(
            vector_store=engine.components.vector_store,
            embedding_extractor=engine.components.embedding_extractor,
        )
        stats = await catalog_store.get_catalog_stats()
    except Exception:
        stats = None

    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        catalog_stats=stats,
    )


@router.get(
    "/v1/intent/catalog",
    response_model=CatalogStatsResponse,
    summary="Get catalog statistics",
    description="Get statistics about the intent catalog.",
)
async def get_catalog_stats(
    _api_key: str = Depends(verify_api_key),
    engine: IntentEngine = Depends(get_engine),
) -> CatalogStatsResponse:
    """
    Get intent catalog statistics.

    Returns counts of examples per intent and category.
    """
    from intent_engine.storage.intent_catalog import IntentCatalogStore

    catalog_store = IntentCatalogStore(
        vector_store=engine.components.vector_store,
        embedding_extractor=engine.components.embedding_extractor,
    )
    stats = await catalog_store.get_catalog_stats()

    return CatalogStatsResponse(
        total_examples=stats["total_examples"],
        num_intents=stats["num_intents"],
        by_intent=stats["by_intent"],
        by_category=stats["by_category"],
    )


@router.get(
    "/v1/intent/intents",
    summary="List core intents",
    description="Get the list of core MVP intents supported by the engine.",
)
async def list_intents(
    _api_key: str = Depends(verify_api_key),
) -> list[dict[str, str]]:
    """
    List the 8 core MVP intents.

    Returns intent codes, categories, and descriptions.
    """
    from intent_engine.storage.intent_catalog import IntentCatalogStore

    # Just get static metadata, no DB needed
    catalog_store = IntentCatalogStore(
        vector_store=None,  # type: ignore
        embedding_extractor=None,
    )
    return catalog_store.get_core_intents()
