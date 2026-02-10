"""Response generation for the orchestration agent."""

import logging
from typing import Any, Protocol

from intent_engine.models.context import CustomerProfile, OrderContext


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt."""
        ...

logger = logging.getLogger(__name__)


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

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """
        Initialize the response generator.

        Args:
            llm_client: Optional LLM client for dynamic response generation.
        """
        self.llm_client = llm_client

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
        # Try LLM generation first if available
        if use_llm and self.llm_client:
            try:
                return await self._generate_with_llm(
                    intent_code, order_context, customer_context, entities
                )
            except Exception as e:
                logger.warning(f"LLM generation failed, using template: {e}")

        # Fall back to template-based generation
        return self._generate_from_template(intent_code, order_context, customer_context)

    async def _generate_with_llm(
        self,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
        entities: list[dict[str, Any]] | None,
    ) -> str:
        """Generate response using LLM."""
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
- Tracking: {order_context.tracking_number or 'Not yet shipped'}
- Carrier: {order_context.carrier or 'N/A'}
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
            status = order_context.status.lower() if isinstance(order_context.status, str) else order_context.status
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
            ctx.update({
                "order_number": f"#{order_context.order_number}",
                "status": order_context.status,
                "carrier": order_context.carrier or "the carrier",
                "tracking_url": order_context.tracking_url or "",
                "tracking_number": order_context.tracking_number or "",
                "currency": order_context.currency,
                "days_until_return_expires": str(order_context.days_until_return_expires or 0),
            })

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
