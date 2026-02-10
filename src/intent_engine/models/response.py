"""Response models for the intent engine."""

from pydantic import BaseModel, Field

from intent_engine.models.entity import ExtractedEntity
from intent_engine.models.intent import ResolvedIntent


class MatchResult(BaseModel):
    """Result from similarity matching against intent catalog."""

    intent_code: str = Field(description="The matched intent code (CATEGORY.INTENT)")
    similarity: float = Field(ge=0.0, le=1.0)
    matched_example: str = Field(description="The catalog example that matched")


class Constraint(BaseModel):
    """A constraint on how an intent should be fulfilled."""

    constraint_type: str  # "deadline", "preference", "policy", "inventory"
    description: str
    value: str
    hard: bool = True  # Hard constraint = must satisfy; soft = prefer


class ReasoningResult(BaseModel):
    """Complete output of the reasoning engine."""

    request_id: str
    resolved_intents: list[ResolvedIntent]
    is_compound: bool = False
    entities: list[ExtractedEntity] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    # Response generation (simplified for MVP)
    customer_response: str | None = None  # Suggested response text
    internal_notes: str | None = None  # For agent dashboard / audit

    # Metadata
    confidence_summary: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence (min of individual intent confidences)",
    )
    requires_human: bool = False
    human_handoff_reason: str | None = None
    reasoning_trace: list[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning log for explainability",
    )
    processing_time_ms: int = 0
    path_taken: str = "fast_path"  # "fast_path" or "reasoning_path"

    model_config = {"json_schema_extra": {"examples": [
        {
            "request_id": "req-12345",
            "resolved_intents": [
                {
                    "category": "ORDER_STATUS",
                    "intent": "WISMO",
                    "confidence": 0.92,
                    "confidence_tier": "high",
                    "evidence": ["where is my order"],
                }
            ],
            "is_compound": False,
            "entities": [
                {
                    "entity_type": "order_id",
                    "value": "ORD-98765",
                    "raw_span": "#ORD-98765",
                    "start_pos": 18,
                    "end_pos": 28,
                    "confidence": 0.99,
                }
            ],
            "constraints": [],
            "confidence_summary": 0.92,
            "requires_human": False,
            "reasoning_trace": ["Fast path match: WISMO (0.92 similarity)"],
            "processing_time_ms": 45,
            "path_taken": "fast_path",
        }
    ]}}
