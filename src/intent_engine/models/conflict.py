"""Conflict resolution models for handling contradictory intents."""

from enum import Enum

from pydantic import BaseModel, Field

from intent_engine.models.intent import ResolvedIntent


class ConflictType(str, Enum):
    """Types of conflicts between intents."""

    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # Return vs Exchange same item
    CONTRADICTORY_POLICY = "contradictory_policy"  # Cancel + Expedite
    POLICY_VIOLATION = "policy_violation"  # Return after window expired


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""

    PREFERENCE = "preference"  # Customer stated preference
    PRIORITY = "priority"  # Applied business rules
    CLARIFICATION = "clarification"  # Need to ask customer
    ESCALATION = "escalation"  # Too complex, send to human


class ConflictResolutionOutput(BaseModel):
    """Output from conflict resolution."""

    resolved_intents: list[ResolvedIntent] = Field(
        description="Final list of intents after conflict resolution"
    )
    has_conflict: bool = Field(
        default=False, description="Whether any conflicts were detected"
    )
    conflict_type: ConflictType | None = Field(
        default=None, description="Type of conflict if detected"
    )
    conflict_description: str | None = Field(
        default=None, description="Human-readable description of the conflict"
    )
    resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.PRIORITY,
        description="Strategy used to resolve the conflict",
    )
    reasoning: list[str] = Field(
        default_factory=list, description="Step-by-step reasoning trace"
    )
    requires_clarification: bool = Field(
        default=False, description="Whether customer input is needed"
    )
    clarification_question: str | None = Field(
        default=None, description="Question to ask customer if clarification needed"
    )
    clarification_options: list[str] = Field(
        default_factory=list, description="Options to present to customer"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "resolved_intents": [
                        {
                            "category": "RETURN_EXCHANGE",
                            "intent": "EXCHANGE_REQUEST",
                            "confidence": 0.85,
                            "confidence_tier": "high",
                            "evidence": ["exchange for a different size"],
                        }
                    ],
                    "has_conflict": True,
                    "conflict_type": "mutually_exclusive",
                    "conflict_description": "Cannot both return and exchange the same item",
                    "resolution_strategy": "preference",
                    "reasoning": [
                        "Detected conflict: RETURN_INITIATE vs EXCHANGE_REQUEST",
                        "Customer preference detected: 'prefer exchange'",
                        "Resolved to EXCHANGE_REQUEST based on stated preference",
                    ],
                    "requires_clarification": False,
                    "clarification_question": None,
                    "clarification_options": [],
                }
            ]
        }
    }
