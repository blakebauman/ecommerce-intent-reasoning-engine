"""Customer service orchestration agent."""

import logging
import time
from typing import Any

from intent_engine.agents.models import (
    ActionType,
    AgentAction,
    AgentResponse,
    ConversationContext,
    CustomerMessage,
)
from intent_engine.agents.response_generator import LLMClient, ResponseGenerator
from intent_engine.config import Settings
from intent_engine.engine import IntentEngine
from intent_engine.integrations.adobe_commerce import (
    AdobeCommerceConnector,
    IMSOAuthAuth,
    IntegrationTokenAuth,
)
from intent_engine.integrations.base import PlatformConnector
from intent_engine.integrations.shopify import ShopifyConnector
from intent_engine.models.context import CustomerProfile, OrderContext
from intent_engine.models.request import InputChannel, IntentRequest


class AnthropicLLMClient:
    """Simple LLM client using Anthropic API directly."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required for LLM features")
        return self._client

    async def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from a prompt."""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


logger = logging.getLogger(__name__)


# Map intents to action types
INTENT_TO_ACTION: dict[str, ActionType] = {
    "ORDER_STATUS.WISMO": ActionType.PROVIDE_ORDER_STATUS,
    "ORDER_STATUS.DELIVERY_ESTIMATE": ActionType.PROVIDE_DELIVERY_ESTIMATE,
    "ORDER_MODIFY.CANCEL_ORDER": ActionType.INITIATE_CANCELLATION,
    "ORDER_MODIFY.CHANGE_ADDRESS": ActionType.UPDATE_SHIPPING_ADDRESS,
    "RETURN_EXCHANGE.RETURN_INITIATE": ActionType.INITIATE_RETURN,
    "RETURN_EXCHANGE.EXCHANGE_REQUEST": ActionType.INITIATE_EXCHANGE,
    "RETURN_EXCHANGE.REFUND_STATUS": ActionType.PROVIDE_REFUND_STATUS,
    "COMPLAINT.DAMAGED_ITEM": ActionType.CREATE_SUPPORT_TICKET,
    # Loyalty and shipping
    "LOYALTY_REWARDS.POINTS_BALANCE": ActionType.PROVIDE_POINTS_BALANCE,
    "LOYALTY_REWARDS.MEMBER_STATUS": ActionType.PROVIDE_MEMBER_STATUS,
    "SHIPPING.SHIPPING_OPTIONS": ActionType.PROVIDE_SHIPPING_OPTIONS,
    "SHIPPING.SHIPPING_COST": ActionType.PROVIDE_SHIPPING_OPTIONS,
    "SHIPPING.INTERNATIONAL": ActionType.PROVIDE_SHIPPING_OPTIONS,
    "SHIPPING.DELIVERY_TIME": ActionType.PROVIDE_SHIPPING_OPTIONS,
}


class CustomerServiceAgent:
    """
    Orchestration agent for customer service interactions.

    Coordinates:
    1. Intent classification via IntentEngine
    2. Order/customer data fetching via platform connectors
    3. Response generation via LLM or templates
    4. Action recommendations based on intent + context
    """

    def __init__(
        self,
        settings: Settings,
        intent_engine: IntentEngine | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """
        Initialize the customer service agent.

        Args:
            settings: Application settings.
            intent_engine: Pre-initialized intent engine (optional).
            llm_client: LLM client for response generation (optional).
        """
        self.settings = settings
        self._intent_engine = intent_engine
        self._llm_client = llm_client
        self._response_generator: ResponseGenerator | None = None
        self._connectors: dict[str, PlatformConnector] = {}
        self._conversations: dict[str, ConversationContext] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent and its components."""
        if self._initialized:
            return

        # Initialize intent engine if not provided
        if self._intent_engine is None:
            self._intent_engine = IntentEngine(settings=self.settings)
            await self._intent_engine.initialize()

        # Initialize LLM client if API key available
        if self._llm_client is None and self.settings.anthropic_api_key:
            self._llm_client = AnthropicLLMClient(
                api_key=self.settings.anthropic_api_key,
                model=self.settings.llm_model,
            )

        # Initialize response generator
        self._response_generator = ResponseGenerator(llm_client=self._llm_client)

        # Initialize platform connectors
        await self._initialize_connectors()

        self._initialized = True
        logger.info("CustomerServiceAgent initialized")

    async def _initialize_connectors(self) -> None:
        """Initialize available platform connectors."""
        # Shopify connector
        if self.settings.shopify_store_domain and self.settings.shopify_access_token:
            self._connectors["shopify"] = ShopifyConnector(
                store_domain=self.settings.shopify_store_domain,
                access_token=self.settings.shopify_access_token,
            )
            logger.info("Shopify connector initialized")

        # Adobe Commerce connector (PaaS)
        if self.settings.adobe_commerce_base_url and self.settings.adobe_commerce_access_token:
            self._connectors["adobe_commerce"] = AdobeCommerceConnector(
                base_url=self.settings.adobe_commerce_base_url,
                auth_strategy=IntegrationTokenAuth(
                    access_token=self.settings.adobe_commerce_access_token
                ),
                store_code=self.settings.adobe_commerce_store_code,
            )
            logger.info("Adobe Commerce connector initialized (PaaS)")

        # Adobe Commerce connector (SaaS) - only if PaaS not configured
        elif (
            self.settings.adobe_commerce_base_url
            and self.settings.adobe_commerce_ims_client_id
            and self.settings.adobe_commerce_ims_client_secret
        ):
            self._connectors["adobe_commerce"] = AdobeCommerceConnector(
                base_url=self.settings.adobe_commerce_base_url,
                auth_strategy=IMSOAuthAuth(
                    client_id=self.settings.adobe_commerce_ims_client_id,
                    client_secret=self.settings.adobe_commerce_ims_client_secret,
                    org_id=self.settings.adobe_commerce_ims_org_id,
                ),
                store_code=self.settings.adobe_commerce_store_code,
            )
            logger.info("Adobe Commerce connector initialized (SaaS)")

    async def shutdown(self) -> None:
        """Shutdown the agent and release resources."""
        if self._intent_engine:
            await self._intent_engine.shutdown()

        for connector in self._connectors.values():
            if hasattr(connector, "close"):
                await connector.close()

        self._initialized = False
        logger.info("CustomerServiceAgent shutdown complete")

    @property
    def intent_engine(self) -> IntentEngine | None:
        """Intent engine used for classification (for lifecycle router)."""
        return self._intent_engine

    def get_connector(self, platform: str | None = None) -> PlatformConnector | None:
        """Get a platform connector."""
        if platform:
            return self._connectors.get(platform)
        # Return first available connector
        if self._connectors:
            return next(iter(self._connectors.values()))
        return None

    async def process_message(self, message: CustomerMessage) -> AgentResponse:
        """
        Process a customer message and generate a response.

        This is the main entry point for the orchestration agent.

        Args:
            message: The customer's message with context.

        Returns:
            AgentResponse with response text and recommended actions.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Get or create conversation context
        context = self._get_or_create_context(message)

        # Step 1: Classify intent
        intent_result = await self._classify_intent(message, context)

        # Step 2: Extract order IDs from entities
        order_ids = self._extract_order_ids(intent_result, message, context)

        # Step 3: Fetch order and customer context
        order_context, customer_context = await self._fetch_context(
            order_ids=order_ids,
            customer_email=message.customer_email,
            platform=message.platform,
        )

        # Step 4: Determine actions
        actions = self._determine_actions(intent_result, order_context, customer_context)

        # Step 5: Generate response
        primary_intent = self._get_primary_intent(intent_result)
        response_text = await self._generate_response(
            intent_code=primary_intent,
            order_context=order_context,
            customer_context=customer_context,
            entities=intent_result.get("entities", []),
            needs_order_id=len(order_ids) == 0 and self._intent_needs_order(primary_intent),
        )

        # Step 6: Update conversation context
        self._update_context(context, intent_result, order_ids)

        # Build response
        processing_time_ms = int((time.time() - start_time) * 1000)

        return AgentResponse(
            message_id=message.message_id,
            conversation_id=context.conversation_id,
            intents=intent_result.get("resolved_intents", []),
            is_compound=intent_result.get("is_compound", False),
            entities=intent_result.get("entities", []),
            response_text=response_text,
            response_tone="empathetic" if "COMPLAINT" in primary_intent else "helpful",
            actions=actions,
            primary_action=actions[0] if actions else None,
            order_context=order_context.model_dump() if order_context else None,
            customer_context=customer_context.model_dump() if customer_context else None,
            requires_human=intent_result.get("requires_human", False),
            human_handoff_reason=intent_result.get("human_handoff_reason"),
            confidence=self._get_confidence(intent_result),
            processing_time_ms=processing_time_ms,
        )

    def _get_or_create_context(self, message: CustomerMessage) -> ConversationContext:
        """Get or create conversation context."""
        conv_id = message.conversation_id or message.message_id

        if conv_id not in self._conversations:
            self._conversations[conv_id] = ConversationContext(
                conversation_id=conv_id,
                customer_email=message.customer_email,
                customer_id=message.customer_id,
                order_ids=list(message.order_ids),
            )

        return self._conversations[conv_id]

    async def _classify_intent(
        self,
        message: CustomerMessage,
        context: ConversationContext,
    ) -> dict[str, Any]:
        """Classify intent using the intent engine."""
        request = IntentRequest(
            request_id=message.message_id,
            tenant_id="agent",
            channel=InputChannel(message.channel)
            if message.channel in ["chat", "email", "voice"]
            else InputChannel.CHAT,
            raw_text=message.text,
            customer_id=message.customer_id,
            customer_tier=context.customer_tier,
            order_ids=context.order_ids,
            previous_intents=context.previous_intents,
        )

        result = await self._intent_engine.resolve(request)

        return {
            "resolved_intents": [
                {
                    "category": intent.category,
                    "intent": intent.intent,
                    "confidence": intent.confidence,
                    "confidence_tier": intent.confidence_tier.value,
                    "evidence": intent.evidence,
                }
                for intent in result.resolved_intents
            ],
            "is_compound": result.is_compound,
            "entities": [
                {
                    "entity_type": entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else entity.entity_type,
                    "value": entity.value,
                    "confidence": entity.confidence,
                }
                for entity in result.entities
            ],
            "requires_human": result.requires_human,
            "human_handoff_reason": result.human_handoff_reason,
            "path_taken": result.path_taken,
        }

    def _extract_order_ids(
        self,
        intent_result: dict[str, Any],
        message: CustomerMessage,
        context: ConversationContext,
    ) -> list[str]:
        """Extract order IDs from entities, message, and context."""
        order_ids = set(context.order_ids)
        order_ids.update(message.order_ids)

        # Extract from entities
        for entity in intent_result.get("entities", []):
            if entity.get("entity_type") == "order_id":
                order_ids.add(entity["value"])

        return list(order_ids)

    async def _fetch_context(
        self,
        order_ids: list[str],
        customer_email: str | None,
        platform: str | None,
    ) -> tuple[OrderContext | None, CustomerProfile | None]:
        """Fetch order and customer context from platform."""
        connector = self.get_connector(platform)
        if not connector:
            return None, None

        order_context = None
        customer_context = None

        # Fetch order context
        if order_ids:
            order_id = order_ids[0]  # Use first order ID
            try:
                if hasattr(connector, "get_order_context_by_number"):
                    order_context = await connector.get_order_context_by_number(order_id)
                if not order_context and hasattr(connector, "get_order_context"):
                    order_context = await connector.get_order_context(order_id)
            except Exception as e:
                logger.warning(f"Failed to fetch order context: {e}")

        # Fetch customer context
        if customer_email:
            try:
                if hasattr(connector, "get_customer_by_email"):
                    customer_context = await connector.get_customer_by_email(customer_email)
            except Exception as e:
                logger.warning(f"Failed to fetch customer context: {e}")

        return order_context, customer_context

    def _determine_actions(
        self,
        intent_result: dict[str, Any],
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
    ) -> list[AgentAction]:
        """Determine actions based on intent and context."""
        actions = []

        for intent in intent_result.get("resolved_intents", []):
            intent_code = f"{intent['category']}.{intent['intent']}"
            action_type = INTENT_TO_ACTION.get(intent_code, ActionType.NONE)

            if action_type == ActionType.NONE:
                continue

            # Build action with context-aware parameters
            action = self._build_action(action_type, intent_code, order_context, customer_context)
            actions.append(action)

        # Add escalation if needed
        if intent_result.get("requires_human"):
            actions.append(
                AgentAction(
                    action_type=ActionType.ESCALATE_TO_HUMAN,
                    description="Escalate to human agent",
                    parameters={
                        "reason": intent_result.get("human_handoff_reason", "Complex request")
                    },
                    requires_confirmation=False,
                )
            )

        return actions

    def _build_action(
        self,
        action_type: ActionType,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
    ) -> AgentAction:
        """Build an action with appropriate parameters."""
        params: dict[str, Any] = {}
        requires_confirmation = False
        description = action_type.value.replace("_", " ").title()

        if order_context:
            params["order_id"] = order_context.order_id
            params["order_number"] = order_context.order_number

        # Action-specific logic
        if action_type == ActionType.PROVIDE_ORDER_STATUS:
            if order_context:
                params["status"] = order_context.status
                params["tracking_number"] = order_context.tracking_number
            description = "Provide order status information"

        elif action_type == ActionType.INITIATE_CANCELLATION:
            requires_confirmation = True
            if order_context:
                can_cancel = order_context.status in ["pending", "processing", "confirmed"]
                params["can_cancel"] = can_cancel
            description = "Initiate order cancellation"

        elif action_type == ActionType.INITIATE_RETURN:
            requires_confirmation = True
            if order_context:
                params["eligible"] = order_context.is_within_return_window
                params["days_remaining"] = order_context.days_until_return_expires
            description = "Initiate return process"

        elif action_type == ActionType.CREATE_SUPPORT_TICKET:
            params["priority"] = (
                "high" if customer_context and customer_context.is_vip else "normal"
            )
            description = "Create support ticket for damaged item"

        return AgentAction(
            action_type=action_type,
            description=description,
            parameters=params,
            requires_confirmation=requires_confirmation,
        )

    def _get_primary_intent(self, intent_result: dict[str, Any]) -> str:
        """Get the primary intent code from results."""
        intents = intent_result.get("resolved_intents", [])
        if not intents:
            return "FALLBACK"

        primary = intents[0]
        return f"{primary['category']}.{primary['intent']}"

    def _intent_needs_order(self, intent_code: str) -> bool:
        """Check if an intent requires an order ID."""
        order_intents = {
            "ORDER_STATUS.WISMO",
            "ORDER_STATUS.DELIVERY_ESTIMATE",
            "ORDER_MODIFY.CANCEL_ORDER",
            "ORDER_MODIFY.CHANGE_ADDRESS",
            "RETURN_EXCHANGE.RETURN_INITIATE",
            "RETURN_EXCHANGE.EXCHANGE_REQUEST",
            "RETURN_EXCHANGE.REFUND_STATUS",
        }
        return intent_code in order_intents

    async def _generate_response(
        self,
        intent_code: str,
        order_context: OrderContext | None,
        customer_context: CustomerProfile | None,
        entities: list[dict[str, Any]],
        needs_order_id: bool = False,
    ) -> str:
        """Generate the response text."""
        if needs_order_id:
            return await self._response_generator.generate(
                "NEEDS_ORDER_ID",
                order_context=None,
                customer_context=customer_context,
            )

        return await self._response_generator.generate(
            intent_code,
            order_context=order_context,
            customer_context=customer_context,
            entities=entities,
        )

    def _get_confidence(self, intent_result: dict[str, Any]) -> float:
        """Get overall confidence from intent result."""
        intents = intent_result.get("resolved_intents", [])
        if not intents:
            return 0.0
        return intents[0].get("confidence", 0.0)

    def _update_context(
        self,
        context: ConversationContext,
        intent_result: dict[str, Any],
        order_ids: list[str],
    ) -> None:
        """Update conversation context with new information."""
        # Update order IDs
        for oid in order_ids:
            if oid not in context.order_ids:
                context.order_ids.append(oid)

        if order_ids:
            context.current_order_id = order_ids[0]

        # Update previous intents
        for intent in intent_result.get("resolved_intents", []):
            intent_code = f"{intent['category']}.{intent['intent']}"
            if intent_code not in context.previous_intents:
                context.previous_intents.append(intent_code)

        context.message_count += 1
