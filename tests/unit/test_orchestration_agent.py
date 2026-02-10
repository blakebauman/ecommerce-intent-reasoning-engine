"""Tests for the customer service orchestration agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.agents.models import ActionType, AgentAction, CustomerMessage
from intent_engine.agents.orchestrator import INTENT_TO_ACTION, CustomerServiceAgent
from intent_engine.agents.response_generator import RESPONSE_TEMPLATES, ResponseGenerator


class TestIntentToActionMapping:
    """Tests for intent to action mapping."""

    def test_wismo_maps_to_provide_status(self) -> None:
        """Test WISMO intent maps to provide order status action."""
        assert INTENT_TO_ACTION["ORDER_STATUS.WISMO"] == ActionType.PROVIDE_ORDER_STATUS

    def test_cancel_maps_to_initiate_cancellation(self) -> None:
        """Test cancel intent maps to cancellation action."""
        assert INTENT_TO_ACTION["ORDER_MODIFY.CANCEL_ORDER"] == ActionType.INITIATE_CANCELLATION

    def test_return_maps_to_initiate_return(self) -> None:
        """Test return intent maps to return action."""
        assert INTENT_TO_ACTION["RETURN_EXCHANGE.RETURN_INITIATE"] == ActionType.INITIATE_RETURN

    def test_damaged_maps_to_support_ticket(self) -> None:
        """Test damaged item maps to support ticket."""
        assert INTENT_TO_ACTION["COMPLAINT.DAMAGED_ITEM"] == ActionType.CREATE_SUPPORT_TICKET


class TestCustomerMessage:
    """Tests for CustomerMessage model."""

    def test_create_minimal_message(self) -> None:
        """Test creating message with minimal fields."""
        msg = CustomerMessage(
            message_id="msg-123",
            text="Where is my order?",
        )
        assert msg.message_id == "msg-123"
        assert msg.text == "Where is my order?"
        assert msg.channel == "chat"
        assert msg.platform is None

    def test_create_full_message(self) -> None:
        """Test creating message with all fields."""
        msg = CustomerMessage(
            message_id="msg-123",
            conversation_id="conv-456",
            customer_email="test@example.com",
            customer_id="cust-789",
            text="Where is my order #12345?",
            channel="email",
            platform="shopify",
            order_ids=["12345"],
            metadata={"source": "test"},
        )
        assert msg.conversation_id == "conv-456"
        assert msg.customer_email == "test@example.com"
        assert msg.platform == "shopify"
        assert "12345" in msg.order_ids


class TestAgentAction:
    """Tests for AgentAction model."""

    def test_create_simple_action(self) -> None:
        """Test creating a simple action."""
        action = AgentAction(
            action_type=ActionType.PROVIDE_ORDER_STATUS,
            description="Provide status",
        )
        assert action.action_type == ActionType.PROVIDE_ORDER_STATUS
        assert action.requires_confirmation is False
        assert action.auto_execute is False

    def test_create_action_requiring_confirmation(self) -> None:
        """Test action that requires confirmation."""
        action = AgentAction(
            action_type=ActionType.INITIATE_CANCELLATION,
            description="Cancel order",
            requires_confirmation=True,
            parameters={"order_id": "12345"},
        )
        assert action.requires_confirmation is True
        assert action.parameters["order_id"] == "12345"


class TestResponseGenerator:
    """Tests for response generation."""

    def test_templates_exist_for_core_intents(self) -> None:
        """Test that templates exist for all core intents."""
        core_intents = [
            "ORDER_STATUS.WISMO",
            "ORDER_STATUS.DELIVERY_ESTIMATE",
            "ORDER_MODIFY.CANCEL_ORDER",
            "ORDER_MODIFY.CHANGE_ADDRESS",
            "RETURN_EXCHANGE.RETURN_INITIATE",
            "RETURN_EXCHANGE.EXCHANGE_REQUEST",
            "RETURN_EXCHANGE.REFUND_STATUS",
            "COMPLAINT.DAMAGED_ITEM",
        ]
        for intent in core_intents:
            assert intent in RESPONSE_TEMPLATES, f"Missing template for {intent}"

    def test_template_has_default_variant(self) -> None:
        """Test that each template has a default variant."""
        for intent, templates in RESPONSE_TEMPLATES.items():
            assert "default" in templates, f"Missing default for {intent}"

    @pytest.mark.asyncio
    async def test_generate_without_llm(self) -> None:
        """Test template-based generation without LLM."""
        generator = ResponseGenerator(llm_client=None)
        response = await generator.generate(
            intent_code="NEEDS_ORDER_ID",
            order_context=None,
            customer_context=None,
        )
        assert "order number" in response.lower()

    @pytest.mark.asyncio
    async def test_generate_fallback_for_unknown_intent(self) -> None:
        """Test fallback for unknown intent."""
        generator = ResponseGenerator(llm_client=None)
        response = await generator.generate(
            intent_code="UNKNOWN.INTENT",
            order_context=None,
            customer_context=None,
        )
        # Should use FALLBACK template
        assert len(response) > 0


class TestCustomerServiceAgent:
    """Tests for the main orchestration agent."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.anthropic_api_key = ""
        settings.llm_model = "claude-sonnet-4-5"
        settings.shopify_store_domain = ""
        settings.shopify_access_token = ""
        settings.adobe_commerce_base_url = ""
        settings.adobe_commerce_access_token = ""
        settings.adobe_commerce_ims_client_id = ""
        settings.adobe_commerce_ims_client_secret = ""
        settings.adobe_commerce_ims_org_id = ""
        settings.adobe_commerce_store_code = "default"
        return settings

    @pytest.fixture
    def mock_engine(self):
        """Create mock intent engine."""
        engine = MagicMock()
        engine.initialize = AsyncMock()
        engine.shutdown = AsyncMock()
        engine.resolve = AsyncMock()
        return engine

    def test_agent_creation(self, mock_settings) -> None:
        """Test agent creation."""
        agent = CustomerServiceAgent(settings=mock_settings)
        assert agent._initialized is False
        assert agent._connectors == {}

    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_settings, mock_engine) -> None:
        """Test agent initialization."""
        agent = CustomerServiceAgent(
            settings=mock_settings,
            intent_engine=mock_engine,
        )

        await agent.initialize()

        assert agent._initialized is True
        assert agent._response_generator is not None

    @pytest.mark.asyncio
    async def test_process_message_basic(self, mock_settings, mock_engine) -> None:
        """Test basic message processing."""
        # Setup mock response
        mock_result = MagicMock()
        mock_result.resolved_intents = [
            MagicMock(
                category="ORDER_STATUS",
                intent="WISMO",
                confidence=0.92,
                confidence_tier=MagicMock(value="high"),
                evidence=["where is my order"],
            )
        ]
        mock_result.is_compound = False
        mock_result.entities = []
        mock_result.requires_human = False
        mock_result.human_handoff_reason = None
        mock_result.path_taken = "fast_path"
        mock_engine.resolve = AsyncMock(return_value=mock_result)

        agent = CustomerServiceAgent(
            settings=mock_settings,
            intent_engine=mock_engine,
        )
        await agent.initialize()

        message = CustomerMessage(
            message_id="msg-123",
            text="Where is my order?",
        )

        response = await agent.process_message(message)

        assert response.message_id == "msg-123"
        assert len(response.intents) == 1
        assert response.intents[0]["category"] == "ORDER_STATUS"
        assert response.processing_time_ms >= 0  # May be 0 in fast tests

    @pytest.mark.asyncio
    async def test_process_message_extracts_order_id(self, mock_settings, mock_engine) -> None:
        """Test that order IDs are extracted from entities."""
        mock_result = MagicMock()
        mock_result.resolved_intents = [
            MagicMock(
                category="ORDER_STATUS",
                intent="WISMO",
                confidence=0.92,
                confidence_tier=MagicMock(value="high"),
                evidence=["where is my order"],
            )
        ]
        mock_result.is_compound = False
        mock_result.entities = [
            MagicMock(entity_type=MagicMock(value="order_id"), value="12345", confidence=0.99)
        ]
        mock_result.requires_human = False
        mock_result.human_handoff_reason = None
        mock_result.path_taken = "fast_path"
        mock_engine.resolve = AsyncMock(return_value=mock_result)

        agent = CustomerServiceAgent(
            settings=mock_settings,
            intent_engine=mock_engine,
        )
        await agent.initialize()

        message = CustomerMessage(
            message_id="msg-123",
            text="Where is my order #12345?",
        )

        response = await agent.process_message(message)

        # Check entities were extracted
        assert any(e["value"] == "12345" for e in response.entities)

    @pytest.mark.asyncio
    async def test_conversation_context_persists(self, mock_settings, mock_engine) -> None:
        """Test that conversation context persists across messages."""
        mock_result = MagicMock()
        mock_result.resolved_intents = [
            MagicMock(
                category="ORDER_STATUS",
                intent="WISMO",
                confidence=0.92,
                confidence_tier=MagicMock(value="high"),
                evidence=[],
            )
        ]
        mock_result.is_compound = False
        mock_result.entities = [
            MagicMock(entity_type=MagicMock(value="order_id"), value="12345", confidence=0.99)
        ]
        mock_result.requires_human = False
        mock_result.human_handoff_reason = None
        mock_result.path_taken = "fast_path"
        mock_engine.resolve = AsyncMock(return_value=mock_result)

        agent = CustomerServiceAgent(
            settings=mock_settings,
            intent_engine=mock_engine,
        )
        await agent.initialize()

        # First message
        msg1 = CustomerMessage(
            message_id="msg-1",
            conversation_id="conv-123",
            text="Where is my order #12345?",
        )
        await agent.process_message(msg1)

        # Check context was created
        assert "conv-123" in agent._conversations
        context = agent._conversations["conv-123"]
        assert "12345" in context.order_ids

        # Second message - order ID should persist
        msg2 = CustomerMessage(
            message_id="msg-2",
            conversation_id="conv-123",
            text="When will it arrive?",
        )
        mock_result.resolved_intents[0].intent = "DELIVERY_ESTIMATE"
        mock_result.entities = []  # No order ID in second message

        await agent.process_message(msg2)

        # Context should still have order ID
        assert "12345" in agent._conversations["conv-123"].order_ids

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_settings, mock_engine) -> None:
        """Test agent shutdown."""
        agent = CustomerServiceAgent(
            settings=mock_settings,
            intent_engine=mock_engine,
        )
        await agent.initialize()
        await agent.shutdown()

        assert agent._initialized is False
        mock_engine.shutdown.assert_called_once()
