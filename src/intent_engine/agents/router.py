"""Lifecycle router: route messages to PrePurchaseAgent or CustomerServiceAgent by intent."""

import logging
import time
from typing import Any

from intent_engine.agents.catalog_agent import get_catalog_provider_from_settings
from intent_engine.agents.models import AgentResponse, CustomerMessage
from intent_engine.agents.pre_purchase_agent import PrePurchaseDeps, get_pre_purchase_agent
from intent_engine.engine import IntentEngine
from intent_engine.models.request import InputChannel, IntentRequest

logger = logging.getLogger(__name__)

# Intent categories that are handled by the pre-purchase agent
PRE_PURCHASE_CATEGORIES = frozenset({"PRODUCT_INQUIRY", "DISCOVERY"})


class LifecycleRouter:
    """
    Routes customer messages to PrePurchaseAgent or CustomerServiceAgent.

    Classifies intent first; if primary intent is PRODUCT_INQUIRY or DISCOVERY,
    delegates to PrePurchaseAgent. Otherwise uses CustomerServiceAgent (post-purchase).
    """

    def __init__(
        self,
        intent_engine: IntentEngine,
        customer_service_agent: Any,  # CustomerServiceAgent
        catalog_provider: Any = None,  # CatalogProvider | None
    ) -> None:
        self.intent_engine = intent_engine
        self.customer_service_agent = customer_service_agent
        self._catalog_provider = catalog_provider
        self._pre_purchase_agent = get_pre_purchase_agent()
        self._pre_purchase_deps: PrePurchaseDeps | None = None

    def _get_pre_purchase_deps(self) -> PrePurchaseDeps:
        if self._pre_purchase_deps is None:
            catalog = self._catalog_provider
            if catalog is None:
                catalog = get_catalog_provider_from_settings()
            self._pre_purchase_deps = PrePurchaseDeps(
                intent_engine=self.intent_engine,
                catalog_provider=catalog,
            )
        return self._pre_purchase_deps

    async def process_message(self, message: CustomerMessage) -> AgentResponse:
        """
        Process a customer message: classify intent, then route to pre-purchase or post-purchase agent.

        Returns:
            AgentResponse with unified shape from either agent.
        """
        start_time = time.time()
        if self.intent_engine is None:
            return await self._run_post_purchase(message, start_time)
        request = IntentRequest(
            request_id=message.message_id,
            tenant_id="router",
            channel=InputChannel(message.channel) if message.channel in ("chat", "email", "voice") else InputChannel.CHAT,
            raw_text=message.text,
            customer_id=message.customer_id,
            order_ids=message.order_ids,
        )
        result = await self.intent_engine.resolve(request)
        primary = result.resolved_intents[0] if result.resolved_intents else None
        category = primary.category if primary else None

        if category in PRE_PURCHASE_CATEGORIES:
            try:
                response = await self._run_pre_purchase(message)
                response.processing_time_ms = int((time.time() - start_time) * 1000)
                return response
            except Exception as e:
                logger.warning("Pre-purchase agent failed, falling back to customer service: %s", e)
                # Fall through to post-purchase agent

        return await self._run_post_purchase(message, start_time)

    async def _run_pre_purchase(self, message: CustomerMessage) -> AgentResponse:
        deps = self._get_pre_purchase_deps()
        run_result = await self._pre_purchase_agent.run(message.text, deps=deps)
        out = run_result.output
        intents: list[dict[str, Any]] = []
        if out.primary_intent:
            intents = [{"category": out.primary_intent.split(".")[0], "intent": out.primary_intent.split(".")[-1], "confidence": out.confidence}]
        return AgentResponse(
            message_id=message.message_id,
            conversation_id=message.conversation_id,
            intents=intents,
            is_compound=False,
            entities=[],
            response_text=out.response_text,
            response_tone="helpful",
            actions=[],
            primary_action=None,
            order_context=None,
            customer_context=None,
            requires_human=False,
            human_handoff_reason=None,
            confidence=out.confidence,
            processing_time_ms=0,
        )

    async def _run_post_purchase(self, message: CustomerMessage, start_time: float | None = None) -> AgentResponse:
        if start_time is None:
            start_time = time.time()
        response = await self.customer_service_agent.process_message(message)
        response.processing_time_ms = int((time.time() - start_time) * 1000)
        return response
