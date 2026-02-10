"""Pydantic AI agent configuration for intent reasoning."""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class DecomposedIntent(BaseModel):
    """A single atomic intent extracted from a compound request."""

    intent_code: str = Field(
        description="The intent code (e.g., ORDER_STATUS.WISMO, RETURN_EXCHANGE.RETURN_INITIATE)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in this classification (0-1)"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Text spans or phrases that support this classification",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Any constraints mentioned (deadlines, preferences, conditions)",
    )


class DecompositionResult(BaseModel):
    """Result of intent decomposition by the LLM."""

    intents: list[DecomposedIntent] = Field(
        description="List of atomic intents found in the message"
    )
    is_compound: bool = Field(
        description="True if multiple distinct intents were found"
    )
    requires_clarification: bool = Field(
        default=False,
        description="True if the intent is genuinely unclear and needs user clarification",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Question to ask if clarification is needed",
    )
    reasoning: str = Field(
        description="Brief explanation of the decomposition reasoning",
    )


@dataclass
class IntentContext:
    """Context passed to the intent decomposition agent."""

    raw_text: str
    extracted_entities: list[dict]
    match_hints: list[str]
    customer_tier: str | None = None
    previous_intents: list[str] | None = None


# Build the core intents documentation for the system prompt
CORE_INTENTS_DOC = """
## Available Intents (MVP - 8 Core Intents)

| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| ORDER_STATUS.WISMO | Where is my order / tracking | "where is my order", "track my package", "order status" |
| ORDER_STATUS.DELIVERY_ESTIMATE | When will it arrive | "when will it arrive", "delivery date", "expected delivery" |
| ORDER_MODIFY.CANCEL_ORDER | Cancel an order | "cancel my order", "I want to cancel", "don't want it anymore" |
| ORDER_MODIFY.CHANGE_ADDRESS | Change shipping address | "change address", "ship to different address", "wrong address" |
| RETURN_EXCHANGE.RETURN_INITIATE | Start a return | "return this", "send it back", "I want to return" |
| RETURN_EXCHANGE.EXCHANGE_REQUEST | Exchange for different item | "exchange for", "swap for different size", "want a different one" |
| RETURN_EXCHANGE.REFUND_STATUS | Check refund status | "where's my refund", "refund status", "when will I get my money back" |
| COMPLAINT.DAMAGED_ITEM | Item arrived damaged | "arrived damaged", "broken", "item is defective" |
"""

SYSTEM_PROMPT = f"""You are an eCommerce intent classification engine. Your job is to analyze customer messages and identify the specific intents they express.

{CORE_INTENTS_DOC}

## Instructions

1. Analyze the customer message carefully
2. Identify ALL distinct intents present (there may be multiple)
3. For each intent, provide:
   - The exact intent code from the table above
   - Your confidence (0.0-1.0) based on how clearly the intent is expressed
   - Evidence: the specific words/phrases that indicate this intent
   - Any constraints mentioned (deadlines, preferences, conditions)
4. If the message contains multiple intents, set is_compound=true
5. If the intent is genuinely unclear or doesn't match any category, set requires_clarification=true

## Guidelines

- Be precise: only use intent codes from the list above
- Be thorough: identify ALL intents, not just the primary one
- Compound example: "I want to return this and get a refund" → RETURN_EXCHANGE.RETURN_INITIATE + RETURN_EXCHANGE.REFUND_STATUS
- Extract constraints like "by Friday", "ASAP", "need it before..."
- If entities like order IDs are mentioned, note them in evidence
"""


def get_intent_agent(model_name: str = "claude-sonnet-4-5") -> Agent[IntentContext, DecompositionResult]:
    """
    Create a Pydantic AI agent for intent decomposition.

    Args:
        model_name: The Anthropic model to use (without provider prefix).

    Returns:
        Configured Agent instance.
    """
    # Use the string-based model identifier format
    model_id = f"anthropic:{model_name}"

    agent: Agent[IntentContext, DecompositionResult] = Agent(
        model_id,
        deps_type=IntentContext,
        output_type=DecompositionResult,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.system_prompt
    def add_context(ctx: RunContext[IntentContext]) -> str:
        """Add dynamic context to the system prompt."""
        context_parts = []

        if ctx.deps.match_hints:
            context_parts.append(
                f"Embedding similarity suggests these intents: {', '.join(ctx.deps.match_hints)}"
            )

        if ctx.deps.extracted_entities:
            entity_str = ", ".join(
                f"{e.get('entity_type')}: {e.get('value')}"
                for e in ctx.deps.extracted_entities
            )
            context_parts.append(f"Extracted entities: {entity_str}")

        if ctx.deps.customer_tier:
            context_parts.append(f"Customer tier: {ctx.deps.customer_tier}")

        if ctx.deps.previous_intents:
            context_parts.append(
                f"Previous intents in conversation: {', '.join(ctx.deps.previous_intents)}"
            )

        if context_parts:
            return "\n## Additional Context\n" + "\n".join(f"- {p}" for p in context_parts)
        return ""

    return agent
