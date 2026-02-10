"""A2A Agent Card - Capabilities advertisement for agent discovery."""

from typing import Any

from pydantic import BaseModel, Field


class ActionSchema(BaseModel):
    """Schema definition for an agent action."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class AgentAction(BaseModel):
    """An action that this agent can perform."""

    name: str
    description: str
    input_schema: ActionSchema
    output_schema: ActionSchema


class AgentCard(BaseModel):
    """
    A2A Agent Card - Describes agent capabilities for discovery.

    This follows the A2A protocol specification for agent advertisement.
    Other agents can fetch this to understand what this agent can do.
    """

    name: str = Field(description="Unique agent identifier")
    description: str = Field(description="Human-readable description")
    version: str = Field(description="Semantic version")
    capabilities: list[str] = Field(
        description="High-level capability tags for filtering"
    )
    actions: list[AgentAction] = Field(description="Available actions")
    url: str | None = Field(default=None, description="Base URL for A2A endpoints")
    documentation_url: str | None = Field(
        default=None, description="Link to documentation"
    )


def get_agent_card(base_url: str | None = None) -> AgentCard:
    """
    Generate the agent card for the intent engine.

    Args:
        base_url: Base URL where A2A endpoints are hosted.

    Returns:
        AgentCard describing this agent's capabilities.
    """
    return AgentCard(
        name="intent-engine",
        description=(
            "eCommerce intent classification and reasoning engine. "
            "Classifies customer messages into intent categories, extracts entities, "
            "and provides confidence-scored results with explainability."
        ),
        version="0.1.0",
        capabilities=[
            "intent-resolution",
            "entity-extraction",
            "ecommerce",
            "customer-support",
        ],
        url=base_url,
        documentation_url="https://github.com/orderloop/intent-engine",
        actions=[
            AgentAction(
                name="resolve_intent",
                description=(
                    "Classify customer message into intent categories with full "
                    "reasoning. Returns resolved intents with confidence scores, "
                    "extracted entities, and processing metadata."
                ),
                input_schema=ActionSchema(
                    type="object",
                    properties={
                        "raw_text": {
                            "type": "string",
                            "description": "The customer message to classify",
                        },
                        "customer_tier": {
                            "type": "string",
                            "enum": ["standard", "VIP"],
                            "description": "Customer tier for priority routing",
                        },
                        "order_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Known order IDs for context",
                        },
                        "previous_intents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Previously resolved intents in conversation",
                        },
                    },
                    required=["raw_text"],
                ),
                output_schema=ActionSchema(
                    type="object",
                    properties={
                        "resolved_intents": {
                            "type": "array",
                            "description": "List of classified intents",
                        },
                        "entities": {
                            "type": "array",
                            "description": "Extracted entities (order IDs, dates, etc.)",
                        },
                        "confidence_summary": {
                            "type": "number",
                            "description": "Overall confidence score (0-1)",
                        },
                        "requires_human": {
                            "type": "boolean",
                            "description": "Whether human handoff is recommended",
                        },
                    },
                    required=["resolved_intents", "confidence_summary"],
                ),
            ),
            AgentAction(
                name="classify_intent_fast",
                description=(
                    "Quick intent classification using embeddings only (no LLM). "
                    "Best for simple, single-intent messages where speed is critical."
                ),
                input_schema=ActionSchema(
                    type="object",
                    properties={
                        "raw_text": {
                            "type": "string",
                            "description": "The customer message to classify",
                        },
                    },
                    required=["raw_text"],
                ),
                output_schema=ActionSchema(
                    type="object",
                    properties={
                        "decision": {
                            "type": "string",
                            "description": "Match decision: fast_path, ambiguous, or low_confidence",
                        },
                        "top_matches": {
                            "type": "array",
                            "description": "Top matching intents with similarity scores",
                        },
                    },
                    required=["decision", "top_matches"],
                ),
            ),
            AgentAction(
                name="list_intent_taxonomy",
                description=(
                    "Return all supported intent categories. Use this to understand "
                    "what kinds of customer intents can be resolved."
                ),
                input_schema=ActionSchema(
                    type="object",
                    properties={},
                    required=[],
                ),
                output_schema=ActionSchema(
                    type="object",
                    properties={
                        "intent_count": {
                            "type": "integer",
                            "description": "Number of supported intents",
                        },
                        "intents": {
                            "type": "array",
                            "description": "List of intent definitions",
                        },
                        "categories": {
                            "type": "array",
                            "description": "High-level intent categories",
                        },
                    },
                    required=["intent_count", "intents"],
                ),
            ),
        ],
    )
