"""Response generation for the orchestration agent."""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from intent_engine.models.context import CustomerProfile, OrderContext


class LLMClient(Protocol):
    """Protocol for LLM clients (legacy interface)."""

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt."""
        ...


logger = logging.getLogger(__name__)


# Pydantic AI structured output for responses
class GeneratedResponse(BaseModel):
    """Structured response from the response generator."""

    text: str = Field(description="The customer-facing response text")
    tone: Literal["empathetic", "helpful", "apologetic", "informative", "urgent"] = Field(
        description="The tone of the response"
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up actions for the agent",
    )
    requires_followup: bool = Field(
        default=False,
        description="Whether the customer needs to provide more information",
    )
    followup_question: str | None = Field(
        default=None,
        description="Question to ask if followup is required",
    )


@dataclass
class ResponseContext:
    """Context for response generation."""

    intent_code: str
    order_number: str | None = None
    order_status: str | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    tracking_url: str | None = None
    estimated_delivery: str | None = None
    is_within_return_window: bool | None = None
    days_until_return_expires: int | None = None
    refund_amount: str | None = None
    customer_name: str | None = None
    customer_tier: str | None = None
    is_vip: bool = False
    entities: list[dict[str, Any]] | None = None


RESPONSE_SYSTEM_PROMPT = """You are a helpful customer service agent for an eCommerce company.
Generate concise, friendly responses based on the customer's intent and available context.

Guidelines:
- Be helpful and empathetic, especially for complaints
- Keep responses concise (2-3 sentences max)
- Include specific order details when available
- If information is missing, politely ask for it
- Match the appropriate tone for the situation:
  - empathetic: for complaints, damaged items, frustration
  - apologetic: for mistakes, delays, issues caused by us
  - helpful: for general inquiries, returns, status checks
  - informative: for product questions, policy explanations
  - urgent: for time-sensitive issues
- For VIP customers, acknowledge their status subtly
- Always be professional and solution-oriented
"""


def get_response_agent(
    model_name: str = "claude-haiku-3-5",
    model: any = None,
) -> Agent[ResponseContext, GeneratedResponse]:
    """
    Create a Pydantic AI agent for response generation.

    Uses Haiku by default for faster/cheaper response generation.

    Args:
        model_name: The Anthropic model to use.
        model: Optional model instance (e.g., TestModel for testing). If provided,
               model_name is ignored.

    Returns:
        Configured Agent instance.
    """
    # Use provided model or construct from model_name
    model_or_id = model if model is not None else f"anthropic:{model_name}"

    agent: Agent[ResponseContext, GeneratedResponse] = Agent(
        model_or_id,
        deps_type=ResponseContext,
        output_type=GeneratedResponse,
        system_prompt=RESPONSE_SYSTEM_PROMPT,
        model_settings=ModelSettings(
            temperature=0.7,  # Slightly higher for natural responses
            max_tokens=512,
        ),
    )

    @agent.system_prompt
    def add_response_context(ctx: RunContext[ResponseContext]) -> str:
        """Add dynamic context for response generation."""
        parts = [f"Customer intent: {ctx.deps.intent_code}"]

        if ctx.deps.order_number:
            parts.append(f"Order: #{ctx.deps.order_number}")
        if ctx.deps.order_status:
            parts.append(f"Status: {ctx.deps.order_status}")
        if ctx.deps.carrier:
            parts.append(f"Carrier: {ctx.deps.carrier}")
        if ctx.deps.tracking_number:
            parts.append(f"Tracking: {ctx.deps.tracking_number}")
        if ctx.deps.tracking_url:
            parts.append(f"Tracking URL: {ctx.deps.tracking_url}")
        if ctx.deps.estimated_delivery:
            parts.append(f"Expected delivery: {ctx.deps.estimated_delivery}")
        if ctx.deps.is_within_return_window is not None:
            parts.append(f"Return eligible: {ctx.deps.is_within_return_window}")
        if ctx.deps.days_until_return_expires is not None:
            parts.append(f"Days until return window expires: {ctx.deps.days_until_return_expires}")
        if ctx.deps.refund_amount:
            parts.append(f"Refund amount: {ctx.deps.refund_amount}")
        if ctx.deps.customer_name:
            parts.append(f"Customer: {ctx.deps.customer_name}")
        if ctx.deps.is_vip:
            parts.append("Customer is VIP")
        if ctx.deps.entities:
            entity_str = ", ".join(f"{e['entity_type']}: {e['value']}" for e in ctx.deps.entities)
            parts.append(f"Extracted entities: {entity_str}")

        return "\n## Context\n" + "\n".join(f"- {p}" for p in parts)

    return agent


# Singleton agent instance
_response_agent: Agent[ResponseContext, GeneratedResponse] | None = None


def get_default_response_agent() -> Agent[ResponseContext, GeneratedResponse]:
    """Get or create the default response agent."""
    global _response_agent
    if _response_agent is None:
        _response_agent = get_response_agent()
    return _response_agent


# Response templates for common intents (fallback when LLM unavailable)
RESPONSE_TEMPLATES = {
    "ORDER_STATUS.WISMO": {
        "shipped": "Your order {order_number} is on its way! It's currently {status} via {carrier}. Track it here: {tracking_url}",
        "in_transit": "Your order {order_number} is in transit with {carrier}. Expected delivery: {estimated_delivery}. Tracking: {tracking_url}",
        "delivered": "Great news! Your order {order_number} was delivered on {delivered_at}.",
        "processing": "Your order {order_number} is being prepared for shipment. We'll send tracking info once it ships.",
        "pending": "Your order {order_number} is confirmed and will be processed shortly.",
        "default": "Your order {order_number} is currently {status}.",
    },
    "ORDER_STATUS.DELIVERY_ESTIMATE": {
        "with_estimate": "Your order {order_number} is expected to arrive by {estimated_delivery}.",
        "shipped": "Your order {order_number} shipped on {shipped_at} and typically takes 3-5 business days.",
        "default": "Your order {order_number} is {status}. Once shipped, delivery typically takes 3-5 business days.",
    },
    "ORDER_MODIFY.CANCEL_ORDER": {
        "can_cancel": "I can help you cancel order {order_number}. Would you like me to proceed with the cancellation?",
        "shipped": "Unfortunately, order {order_number} has already shipped and cannot be cancelled. Would you like to initiate a return instead?",
        "delivered": "Order {order_number} has already been delivered. I can help you start a return if needed.",
        "default": "Let me check the cancellation options for order {order_number}.",
    },
    "ORDER_MODIFY.CHANGE_ADDRESS": {
        "can_change": "I can update the shipping address for order {order_number}. What's the new address?",
        "shipped": "Order {order_number} has already shipped, so we can't change the address. You may be able to redirect it through {carrier}'s website.",
        "default": "Let me check if we can update the address for order {order_number}.",
    },
    "RETURN_EXCHANGE.RETURN_INITIATE": {
        "eligible": "I can help you return items from order {order_number}. You have {days_until_return_expires} days left in your return window. Which items would you like to return?",
        "expired": "I'm sorry, but the return window for order {order_number} has expired. It ended on {return_window_ends}. Would you like me to connect you with a specialist who might be able to help?",
        "default": "Let me check the return eligibility for order {order_number}.",
    },
    "RETURN_EXCHANGE.EXCHANGE_REQUEST": {
        "eligible": "I can help you exchange items from order {order_number}. What would you like to exchange and for what size/color?",
        "default": "Let me look into exchange options for order {order_number}.",
    },
    "RETURN_EXCHANGE.REFUND_STATUS": {
        "refunded": "Your refund of {refund_amount} {currency} for order {order_number} has been processed. It should appear in your account within 5-10 business days.",
        "pending": "Your return for order {order_number} is being processed. Once we receive and inspect the items, your refund will be issued within 3-5 business days.",
        "default": "Let me check the refund status for order {order_number}.",
    },
    "COMPLAINT.DAMAGED_ITEM": {
        "default": "I'm so sorry to hear your item arrived damaged. I'd like to make this right. Can you tell me which item was damaged and send a photo if possible? I can arrange a replacement or refund.",
    },
    "LOYALTY_REWARDS.POINTS_BALANCE": {
        "default": "I can look up your rewards balance. One moment—your points balance will appear in your account dashboard, or I can have a specialist share it with you.",
    },
    "LOYALTY_REWARDS.MEMBER_STATUS": {
        "default": "I'd be happy to help with your member status. You can check your tier and benefits in your account, or I can connect you with someone who can confirm your status.",
    },
    "LOYALTY_REWARDS.REDEEM_POINTS": {
        "default": "You can redeem your points at checkout or in the rewards section of your account. If you tell me what you're interested in, I can point you to the right place.",
    },
    "SHIPPING.SHIPPING_OPTIONS": {
        "default": "We offer standard and express shipping. Standard typically arrives in 5–7 business days; express in 2–3. Options and costs are shown at checkout.",
    },
    "SHIPPING.SHIPPING_COST": {
        "default": "Shipping cost depends on your location and the option you choose. You'll see the exact amount at checkout before you pay.",
    },
    "SHIPPING.INTERNATIONAL": {
        "default": "We do ship internationally to many countries. At checkout you can enter your address to see availability and shipping costs for your location.",
    },
    "SHIPPING.DELIVERY_TIME": {
        "default": "Standard shipping typically arrives in 5–7 business days; express in 2–3. Exact delivery dates are shown at checkout based on your address.",
    },
    "PRODUCT_INQUIRY.STOCK": {
        "default": "For current stock and product details, please use our product assistant or check the product page—it shows live availability.",
    },
    "DISCOVERY.RECOMMENDATION": {
        "default": "Our product assistant can suggest items based on what you like. Try asking it for recommendations, or browse our best sellers and new arrivals on the site.",
    },
    "NEEDS_ORDER_ID": {
        "default": "I'd be happy to help! Could you please provide your order number? You can find it in your confirmation email.",
    },
    "NEEDS_CLARIFICATION": {
        "default": "I want to make sure I help you with the right thing. Could you please provide more details about what you need?",
    },
    "FALLBACK": {
        "default": "Thank you for reaching out. Let me look into this for you.",
    },
}


class ResponseGenerator:
    """Generates customer-facing responses based on intent and context."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        use_pydantic_ai: bool = True,
    ) -> None:
        """
        Initialize the response generator.

        Args:
            llm_client: Optional legacy LLM client for dynamic response generation.
            use_pydantic_ai: Whether to use Pydantic AI agent (preferred).
        """
        self.llm_client = llm_client
        self.use_pydantic_ai = use_pydantic_ai
        self._agent: Agent[ResponseContext, GeneratedResponse] | None = None

    @property
    def agent(self) -> Agent[ResponseContext, GeneratedResponse]:
        """Lazy-load the Pydantic AI agent."""
        if self._agent is None:
            self._agent = get_default_response_agent()
        return self._agent

    async def generate(
        self,
        intent_code: str,
        order_context: OrderContext | None = None,
        customer_context: CustomerProfile | None = None,
        entities: list[dict[str, Any]] | None = None,
        use_llm: bool = True,
    ) -> str:
        """
        Generate a response for the given intent and context.

        Args:
            intent_code: The resolved intent (e.g., "ORDER_STATUS.WISMO").
            order_context: Order data if available.
            customer_context: Customer profile if available.
            entities: Extracted entities from the message.
            use_llm: Whether to use LLM for generation (falls back to templates).

        Returns:
            Generated response text.
        """
        # Try Pydantic AI agent first (preferred)
        if use_llm and self.use_pydantic_ai:
            try:
                return await self._generate_with_pydantic_ai(
                    intent_code, order_context, customer_context, entities
                )
            except Exception as e:
                logger.warning(f"Pydantic AI generation failed: {e}")

        # Try legacy LLM client if available
        if use_llm and self.llm_client:
            try:
                return await self._generate_with_llm(
                    intent_code, order_context, customer_context, entities
                )
            except Exception as e:
                logger.warning(f"LLM generation failed, using template: {e}")

        # Fall back to template-based generation
        return self._generate_from_template(intent_code, order_context, customer_context)

    async def generate_structured(
        self,
        intent_code: str,
        order_context: OrderContext | None = None,
        customer_context: CustomerProfile | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> GeneratedResponse:
        """
        Generate a structured response with tone and suggested actions.

        Args:
            intent_code: The resolved intent.
            order_context: Order data if available.
            customer_context: Customer profile if available.
            entities: Extracted entities from the message.

        Returns:
            GeneratedResponse with text, tone, and suggested actions.
        """
        ctx = self._build_response_context(intent_code, order_context, customer_context, entities)

        # Build prompt for the agent
        prompt = f"Generate a response for intent: {intent_code}"
        if order_context:
            prompt += f" regarding order #{order_context.order_number}"

        result = await self.agent.run(prompt, deps=ctx)
        return result.output

    async def generate_streaming(
        self,
        intent_code: str,
        order_context: OrderContext | None = None,
        customer_context: CustomerProfile | None = None,
        entities: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """
        Generate a response with streaming output.

        Yields text chunks as they're generated, useful for real-time display.

        Args:
            intent_code: The resolved intent.
            order_context: Order data if available.
            customer_context: Customer profile if available.
            entities: Extracted entities from the message.

        Yields:
            Text chunks as they're generated.
        """
        ctx = self._build_response_context(intent_code, order_context, customer_context, entities)

        # Build prompt for the agent
        prompt = f"Generate a response for intent: {intent_code}"
        if order_context:
            prompt += f" regarding order #{order_context.order_number}"

        async with self.agent.run_stream(prompt, deps=ctx) as result:
            async for chunk in result.stream_text():
                yield chunk

    async def _generate_with_pydantic_ai(
        self,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
        entities: list[dict[str, Any]] | None,
    ) -> str:
        """Generate response using Pydantic AI agent."""
        response = await self.generate_structured(
            intent_code, order_context, customer_context, entities
        )
        return response.text

    def _build_response_context(
        self,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
        entities: list[dict[str, Any]] | None,
    ) -> ResponseContext:
        """Build ResponseContext from order and customer data."""
        ctx = ResponseContext(intent_code=intent_code, entities=entities)

        if order_context:
            ctx.order_number = order_context.order_number
            ctx.order_status = order_context.status
            ctx.carrier = order_context.carrier
            ctx.tracking_number = order_context.tracking_number
            ctx.tracking_url = order_context.tracking_url
            ctx.is_within_return_window = order_context.is_within_return_window
            ctx.days_until_return_expires = order_context.days_until_return_expires
            if order_context.estimated_delivery:
                ctx.estimated_delivery = order_context.estimated_delivery.strftime("%B %d, %Y")
            if order_context.refund_amount:
                ctx.refund_amount = f"{order_context.refund_amount:.2f} {order_context.currency}"

        if customer_context:
            ctx.customer_name = customer_context.name
            ctx.customer_tier = customer_context.tier.value if customer_context.tier else None
            ctx.is_vip = customer_context.is_vip

        return ctx

    async def _generate_with_llm(
        self,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
        entities: list[dict[str, Any]] | None,
    ) -> str:
        """Generate response using legacy LLM client."""
        # Build context for LLM
        context_parts = [f"Customer intent: {intent_code}"]

        if order_context:
            context_parts.append(f"""
Order Information:
- Order Number: {order_context.order_number}
- Status: {order_context.status}
- Fulfillment: {order_context.fulfillment_status}
- Total: {order_context.total} {order_context.currency}
- Created: {order_context.created_at}
- Tracking: {order_context.tracking_number or "Not yet shipped"}
- Carrier: {order_context.carrier or "N/A"}
- Return eligible: {order_context.is_within_return_window}
- Days until return expires: {order_context.days_until_return_expires}
""")

        if customer_context:
            context_parts.append(f"""
Customer Information:
- Name: {customer_context.name}
- Tier: {customer_context.tier.value}
- Total Orders: {customer_context.total_orders}
- Lifetime Value: ${customer_context.lifetime_value:.2f}
""")

        if entities:
            entity_str = ", ".join(f"{e['entity_type']}: {e['value']}" for e in entities)
            context_parts.append(f"Extracted entities: {entity_str}")

        context = "\n".join(context_parts)

        prompt = f"""You are a helpful customer service agent. Generate a concise, friendly response based on this context:

{context}

Guidelines:
- Be helpful and empathetic
- Keep responses concise (2-3 sentences max)
- Include specific order details when available
- If you need more information, ask politely
- Match the customer's tone

Generate only the response text, no explanations."""

        response = await self.llm_client.generate(prompt, max_tokens=256)
        return response.strip()

    def _generate_from_template(
        self,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
    ) -> str:
        """Generate response using templates."""
        templates = RESPONSE_TEMPLATES.get(intent_code, RESPONSE_TEMPLATES["FALLBACK"])

        # Select appropriate template variant
        template_key = "default"
        if order_context:
            status = (
                order_context.status.lower()
                if isinstance(order_context.status, str)
                else order_context.status
            )
            if status in templates:
                template_key = status
            elif order_context.is_within_return_window and "eligible" in templates:
                template_key = "eligible"
            elif not order_context.is_within_return_window and "expired" in templates:
                template_key = "expired"

        template = templates.get(template_key, templates["default"])

        # Build template context
        format_context = self._build_format_context(order_context, customer_context)

        try:
            return template.format(**format_context)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            return templates["default"].format(**format_context)

    def _build_format_context(
        self,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
    ) -> dict[str, Any]:
        """Build context dict for template formatting."""
        ctx: dict[str, Any] = {
            "order_number": "your order",
            "status": "being processed",
            "carrier": "the carrier",
            "tracking_url": "",
            "tracking_number": "",
            "estimated_delivery": "soon",
            "delivered_at": "",
            "shipped_at": "",
            "refund_amount": "",
            "currency": "USD",
            "days_until_return_expires": "30",
            "return_window_ends": "",
            "customer_name": "there",
        }

        if order_context:
            ctx.update(
                {
                    "order_number": f"#{order_context.order_number}",
                    "status": order_context.status,
                    "carrier": order_context.carrier or "the carrier",
                    "tracking_url": order_context.tracking_url or "",
                    "tracking_number": order_context.tracking_number or "",
                    "currency": order_context.currency,
                    "days_until_return_expires": str(order_context.days_until_return_expires or 0),
                }
            )

            if order_context.delivered_at:
                ctx["delivered_at"] = order_context.delivered_at.strftime("%B %d, %Y")
            if order_context.shipped_at:
                ctx["shipped_at"] = order_context.shipped_at.strftime("%B %d, %Y")
            if order_context.return_window_ends:
                ctx["return_window_ends"] = order_context.return_window_ends.strftime("%B %d, %Y")
            if order_context.refund_amount:
                ctx["refund_amount"] = f"{order_context.refund_amount:.2f}"
            if order_context.estimated_delivery:
                ctx["estimated_delivery"] = order_context.estimated_delivery.strftime("%B %d, %Y")

        if customer_context:
            ctx["customer_name"] = customer_context.name or "there"

        return ctx
