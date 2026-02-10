"""WebSocket message models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WSMessageType(str, Enum):
    """WebSocket message types."""

    # Client → Server
    RESOLVE = "resolve"
    PING = "ping"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Server → Client
    CONNECTED = "connected"
    PONG = "pong"
    REASONING_STEP = "reasoning_step"
    RESULT = "result"
    ERROR = "error"
    JOB_UPDATE = "job_update"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class WSMessage(BaseModel):
    """Base WebSocket message."""

    type: WSMessageType
    request_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Client → Server Messages


class ResolveRequest(BaseModel):
    """Request to resolve an intent."""

    raw_text: str
    tenant_id: str | None = None  # Optional, can be derived from auth
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubscribeRequest(BaseModel):
    """Request to subscribe to job updates."""

    job_id: str


# Server → Client Messages


class ConnectedPayload(BaseModel):
    """Payload for connected message."""

    connection_id: str
    tenant_id: str
    server_time: datetime = Field(default_factory=datetime.utcnow)


class ReasoningStepPayload(BaseModel):
    """Payload for reasoning step updates."""

    step_name: str
    description: str
    duration_ms: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorPayload(BaseModel):
    """Payload for error messages."""

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class JobUpdatePayload(BaseModel):
    """Payload for job status updates."""

    job_id: str
    status: str
    progress: float | None = None  # 0.0 to 1.0
    message: str | None = None


# Predefined reasoning steps


class ReasoningStep(str, Enum):
    """Named reasoning steps for streaming updates."""

    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    EMBEDDING_GENERATION = "embedding_generation"
    CONTEXT_ENRICHMENT = "context_enrichment"
    SIMILARITY_MATCHING = "similarity_matching"
    COMPOUND_DETECTION = "compound_detection"
    POLICY_EVALUATION = "policy_evaluation"
    LLM_DECOMPOSITION = "llm_decomposition"
    FAST_PATH = "fast_path"
    REASONING_PATH = "reasoning_path"
    COMPLETE = "complete"


# Step descriptions for UI
STEP_DESCRIPTIONS = {
    ReasoningStep.ENTITY_EXTRACTION: "Extracting entities (orders, products, dates)...",
    ReasoningStep.SENTIMENT_ANALYSIS: "Analyzing customer sentiment...",
    ReasoningStep.EMBEDDING_GENERATION: "Generating semantic embeddings...",
    ReasoningStep.CONTEXT_ENRICHMENT: "Enriching with order and customer context...",
    ReasoningStep.SIMILARITY_MATCHING: "Matching against known intents...",
    ReasoningStep.COMPOUND_DETECTION: "Checking for multiple intents...",
    ReasoningStep.POLICY_EVALUATION: "Evaluating business policies...",
    ReasoningStep.LLM_DECOMPOSITION: "Using AI to decompose complex request...",
    ReasoningStep.FAST_PATH: "High confidence match found, using fast path",
    ReasoningStep.REASONING_PATH: "Using reasoning path for complex resolution",
    ReasoningStep.COMPLETE: "Resolution complete",
}
