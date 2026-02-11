"""Pydantic AI agent configuration for intent reasoning."""

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.settings import ModelSettings


class DecomposedIntent(BaseModel):
    """A single atomic intent extracted from a compound request."""

    intent_code: str = Field(
        description="The intent code (e.g., ORDER_STATUS.WISMO, RETURN_EXCHANGE.RETURN_INITIATE)"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this classification (0-1)")
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
    is_compound: bool = Field(description="True if multiple distinct intents were found")
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
    # Sentiment scores for compound decision-making
    frustration_score: float | None = None
    urgency_score: float | None = None
    sentiment_score: float | None = None
    # Optional order lookup callback for tools
    order_lookup: any = None  # Callable[[str], Awaitable[dict | None]]
    # Optional return eligibility check callback
    return_eligibility_check: any = None  # Callable[[str], Awaitable[dict | None]]


# Build the core intents documentation for the system prompt
CORE_INTENTS_DOC = """
## Available Intents

### Order Status
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| ORDER_STATUS.WISMO | Where is my order / tracking | "where is my order", "track my package", "order status" |
| ORDER_STATUS.DELIVERY_ESTIMATE | When will it arrive | "when will it arrive", "delivery date", "expected delivery" |
| ORDER_STATUS.TRACKING_ISSUE | Problem with tracking | "tracking doesn't work", "tracking stuck", "invalid tracking" |

### Order Modifications
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| ORDER_MODIFY.CANCEL_ORDER | Cancel an order | "cancel my order", "I want to cancel", "don't want it anymore" |
| ORDER_MODIFY.CHANGE_ADDRESS | Change shipping address | "change address", "ship to different address", "wrong address" |
| ORDER_MODIFY.EXPEDITE | Speed up delivery | "rush my order", "expedite shipping", "faster delivery" |

### Returns & Exchanges
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| RETURN_EXCHANGE.RETURN_INITIATE | Start a return | "return this", "send it back", "I want to return" |
| RETURN_EXCHANGE.RETURN_STATUS | Check return status | "did you get my return", "return package status" |
| RETURN_EXCHANGE.EXCHANGE_REQUEST | Exchange for different item | "exchange for", "swap for different size", "want a different one" |
| RETURN_EXCHANGE.REFUND_STATUS | Check refund status | "where's my refund", "refund status", "when will I get my money back" |
| RETURN_EXCHANGE.WARRANTY_CLAIM | Warranty issue | "under warranty", "warranty claim", "covered by warranty" |
| RETURN_EXCHANGE.STORE_CREDIT | Store credit inquiry | "store credit balance", "use store credit" |

### Complaints
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| COMPLAINT.DAMAGED_ITEM | Item arrived damaged | "arrived damaged", "broken", "item is defective" |
| COMPLAINT.WRONG_ITEM | Received wrong item | "wrong item", "not what I ordered" |
| COMPLAINT.MISSING_ITEM | Item missing from order | "missing from package", "item not included" |
| COMPLAINT.QUALITY_ISSUE | Quality problem | "poor quality", "not as described", "looks different" |

### Loyalty & Rewards (NEW)
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| LOYALTY_REWARDS.POINTS_BALANCE | Check points balance | "how many points", "rewards balance" |
| LOYALTY_REWARDS.REDEEM_POINTS | Redeem points | "use my points", "redeem rewards" |
| LOYALTY_REWARDS.MEMBER_STATUS | Member tier status | "what tier am I", "VIP status" |

### Shipping (NEW)
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| SHIPPING.SHIPPING_OPTIONS | Available shipping methods | "shipping options", "delivery methods" |
| SHIPPING.INTERNATIONAL | International shipping | "ship internationally", "ship to UK" |
| SHIPPING.SHIPPING_COST | Shipping cost inquiry | "how much is shipping", "delivery fee" |

### Bulk & Wholesale (NEW)
| Intent Code | Description | Example Phrases |
|-------------|-------------|-----------------|
| BULK_WHOLESALE.BULK_DISCOUNT | Bulk discount inquiry | "bulk discount", "volume pricing" |
| BULK_WHOLESALE.BUSINESS_ACCOUNT | Business account setup | "business account", "B2B ordering" |
| BULK_WHOLESALE.WHOLESALE_PRICING | Wholesale pricing | "wholesale rates", "reseller pricing" |
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

# Valid intent codes for validation
VALID_INTENT_CODES: set[str] = {
    # Order Status
    "ORDER_STATUS.WISMO",
    "ORDER_STATUS.DELIVERY_ESTIMATE",
    "ORDER_STATUS.CONFIRMATION",
    "ORDER_STATUS.TRACKING_ISSUE",
    # Order Modify
    "ORDER_MODIFY.CANCEL_ORDER",
    "ORDER_MODIFY.CHANGE_ADDRESS",
    "ORDER_MODIFY.CHANGE_ITEMS",
    "ORDER_MODIFY.EXPEDITE",
    # Return/Exchange
    "RETURN_EXCHANGE.RETURN_INITIATE",
    "RETURN_EXCHANGE.EXCHANGE_REQUEST",
    "RETURN_EXCHANGE.REFUND_STATUS",
    "RETURN_EXCHANGE.RETURN_STATUS",
    "RETURN_EXCHANGE.RETURN_POLICY",
    "RETURN_EXCHANGE.WARRANTY_CLAIM",
    "RETURN_EXCHANGE.STORE_CREDIT",
    # Complaints
    "COMPLAINT.DAMAGED_ITEM",
    "COMPLAINT.WRONG_ITEM",
    "COMPLAINT.MISSING_ITEM",
    "COMPLAINT.QUALITY_ISSUE",
    "COMPLAINT.LATE_DELIVERY",
    "COMPLAINT.SERVICE_ISSUE",
    "COMPLAINT.LOST_PACKAGE",
    # Product Inquiry
    "PRODUCT_INQUIRY.STOCK",
    "PRODUCT_INQUIRY.RESTOCK",
    "PRODUCT_INQUIRY.SIZE_FIT",
    "PRODUCT_INQUIRY.FEATURES",
    "PRODUCT_INQUIRY.COMPATIBILITY",
    "PRODUCT_INQUIRY.MATERIALS",
    "PRODUCT_INQUIRY.USAGE",
    # Account/Billing
    "ACCOUNT_BILLING.PAYMENT_ISSUE",
    "ACCOUNT_BILLING.UPDATE_PAYMENT",
    "ACCOUNT_BILLING.INVOICE",
    "ACCOUNT_BILLING.PROMO_CODE",
    "ACCOUNT_BILLING.SUBSCRIPTION",
    "ACCOUNT_BILLING.ACCOUNT_ACCESS",
    "ACCOUNT_BILLING.UPDATE_INFO",
    "ACCOUNT_BILLING.DELETE_ACCOUNT",
    # Discovery
    "DISCOVERY.RECOMMENDATION",
    "DISCOVERY.COMPARISON",
    "DISCOVERY.BEST_SELLER",
    "DISCOVERY.NEW_ARRIVALS",
    "DISCOVERY.GIFT_SUGGESTION",
    # Meta
    "META.GREETING",
    "META.FAREWELL",
    "META.HUMAN_HANDOFF",
    "META.FEEDBACK",
    "META.UNCLEAR",
    "META.OFF_TOPIC",
    # Loyalty/Rewards
    "LOYALTY_REWARDS.POINTS_BALANCE",
    "LOYALTY_REWARDS.EARN_POINTS",
    "LOYALTY_REWARDS.REDEEM_POINTS",
    "LOYALTY_REWARDS.MEMBER_STATUS",
    "LOYALTY_REWARDS.POINTS_VALUE",
    # Shipping
    "SHIPPING.SHIPPING_OPTIONS",
    "SHIPPING.INTERNATIONAL",
    "SHIPPING.SHIPPING_COST",
    "SHIPPING.DELIVERY_LOCATIONS",
    "SHIPPING.DELIVERY_TIME",
    # Bulk/Wholesale
    "BULK_WHOLESALE.BULK_DISCOUNT",
    "BULK_WHOLESALE.BUSINESS_ACCOUNT",
    "BULK_WHOLESALE.MINIMUM_ORDER",
    "BULK_WHOLESALE.WHOLESALE_PRICING",
}


def get_intent_agent(
    model_name: str = "claude-sonnet-4-5",
    temperature: float = 0.1,
    max_retries: int = 2,
    model: any = None,
) -> Agent[IntentContext, DecompositionResult]:
    """
    Create a Pydantic AI agent for intent decomposition.

    Args:
        model_name: The Anthropic model to use (without provider prefix).
        temperature: Model temperature (lower = more deterministic). Default 0.1 for classification.
        max_retries: Number of retries on validation failure.
        model: Optional model instance (e.g., TestModel for testing). If provided,
               model_name is ignored.

    Returns:
        Configured Agent instance.
    """
    # Use provided model or construct from model_name
    model_or_id = model if model is not None else f"anthropic:{model_name}"

    agent: Agent[IntentContext, DecompositionResult] = Agent(
        model_or_id,
        deps_type=IntentContext,
        output_type=DecompositionResult,
        system_prompt=SYSTEM_PROMPT,
        retries=max_retries,
        model_settings=ModelSettings(
            temperature=temperature,
            max_tokens=1024,
        ),
    )

    @agent.output_validator
    async def validate_intent_codes(
        ctx: RunContext[IntentContext], result: DecompositionResult
    ) -> DecompositionResult:
        """Validate that all intent codes are valid."""
        invalid_codes = []
        for intent in result.intents:
            if intent.intent_code not in VALID_INTENT_CODES:
                invalid_codes.append(intent.intent_code)

        if invalid_codes:
            valid_list = ", ".join(sorted(VALID_INTENT_CODES)[:10]) + "..."
            raise ModelRetry(
                f"Invalid intent code(s): {invalid_codes}. Use only valid codes like: {valid_list}"
            )

        return result

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
                f"{e.get('entity_type')}: {e.get('value')}" for e in ctx.deps.extracted_entities
            )
            context_parts.append(f"Extracted entities: {entity_str}")

        if ctx.deps.customer_tier:
            context_parts.append(f"Customer tier: {ctx.deps.customer_tier}")

        if ctx.deps.previous_intents:
            context_parts.append(
                f"Previous intents in conversation: {', '.join(ctx.deps.previous_intents)}"
            )

        # Add sentiment context for compound decision-making
        if ctx.deps.frustration_score is not None or ctx.deps.urgency_score is not None:
            sentiment_parts = []
            if ctx.deps.frustration_score is not None:
                sentiment_parts.append(f"frustration={ctx.deps.frustration_score:.2f}")
            if ctx.deps.urgency_score is not None:
                sentiment_parts.append(f"urgency={ctx.deps.urgency_score:.2f}")
            if ctx.deps.sentiment_score is not None:
                sentiment_parts.append(f"sentiment={ctx.deps.sentiment_score:.2f}")
            context_parts.append(f"Sentiment: {', '.join(sentiment_parts)}")

        if context_parts:
            return "\n## Additional Context\n" + "\n".join(f"- {p}" for p in context_parts)
        return ""

    @agent.tool
    async def lookup_order(
        ctx: RunContext[IntentContext], order_id: str
    ) -> dict[str, str | bool | None]:
        """
        Look up order details to help classify the intent.

        Use this tool when the customer mentions an order ID and you need
        more context to determine their intent (e.g., check if order is
        delivered to classify as WISMO vs LATE_DELIVERY complaint).

        Args:
            order_id: The order ID to look up.

        Returns:
            Order details including status, tracking, and eligibility info.
        """
        if ctx.deps.order_lookup is None:
            return {"error": "Order lookup not available", "order_id": order_id}

        try:
            order_data = await ctx.deps.order_lookup(order_id)
            if order_data is None:
                return {"error": "Order not found", "order_id": order_id}
            return order_data
        except Exception as e:
            return {"error": str(e), "order_id": order_id}

    @agent.tool
    async def check_return_eligibility(
        ctx: RunContext[IntentContext], order_id: str
    ) -> dict[str, str | bool | int | None]:
        """
        Check if an order is eligible for return.

        Use this tool when the customer wants to return something and you
        need to verify eligibility before classifying as RETURN_INITIATE.

        Args:
            order_id: The order ID to check.

        Returns:
            Return eligibility info including window status and days remaining.
        """
        if ctx.deps.return_eligibility_check is None:
            return {"error": "Return eligibility check not available", "order_id": order_id}

        try:
            eligibility = await ctx.deps.return_eligibility_check(order_id)
            if eligibility is None:
                return {"error": "Order not found", "order_id": order_id}
            return eligibility
        except Exception as e:
            return {"error": str(e), "order_id": order_id}

    return agent
