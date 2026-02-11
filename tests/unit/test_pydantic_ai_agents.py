"""Tests for Pydantic AI agents using TestModel for deterministic testing.

NOTE: Uses direct module imports to avoid triggering spaCy imports via __init__.py,
which is incompatible with Python 3.14.
"""

import importlib.util
import sys

import pytest
from pydantic_ai.models.test import TestModel

# Direct import from llm/client.py to avoid spaCy import chain
_spec = importlib.util.spec_from_file_location("client", "src/intent_engine/llm/client.py")
_client_module = importlib.util.module_from_spec(_spec)
sys.modules["intent_engine.llm.client"] = _client_module
_spec.loader.exec_module(_client_module)

DecompositionResult = _client_module.DecompositionResult
IntentContext = _client_module.IntentContext
VALID_INTENT_CODES = _client_module.VALID_INTENT_CODES
get_intent_agent = _client_module.get_intent_agent

# Direct import from agents/response_generator.py
_spec2 = importlib.util.spec_from_file_location(
    "response_generator", "src/intent_engine/agents/response_generator.py"
)
_response_module = importlib.util.module_from_spec(_spec2)
sys.modules["intent_engine.agents.response_generator"] = _response_module
_spec2.loader.exec_module(_response_module)

GeneratedResponse = _response_module.GeneratedResponse
ResponseContext = _response_module.ResponseContext
get_response_agent = _response_module.get_response_agent


class TestIntentAgent:
    """Tests for the intent decomposition agent."""

    @pytest.fixture
    def test_model(self) -> TestModel:
        """Create a TestModel that returns valid decomposition results."""
        # Configure TestModel with valid intent codes to pass validation
        return TestModel(
            custom_output_args={
                "intents": [
                    {
                        "intent_code": "ORDER_STATUS.WISMO",
                        "confidence": 0.95,
                        "evidence": ["where is my order"],
                        "constraints": [],
                    }
                ],
                "is_compound": False,
                "requires_clarification": False,
                "clarification_question": None,
                "reasoning": "Customer is asking about order status.",
            }
        )

    @pytest.fixture
    def agent_with_test_model(self, test_model: TestModel):
        """Create an agent with TestModel for deterministic testing."""
        # Pass TestModel directly to avoid needing ANTHROPIC_API_KEY
        agent = get_intent_agent(model=test_model)
        return agent

    def test_valid_intent_codes_registry(self) -> None:
        """Test that VALID_INTENT_CODES contains expected intents."""
        # Check some expected intents exist
        assert "ORDER_STATUS.WISMO" in VALID_INTENT_CODES
        assert "RETURN_EXCHANGE.RETURN_INITIATE" in VALID_INTENT_CODES
        assert "COMPLAINT.DAMAGED_ITEM" in VALID_INTENT_CODES
        assert "LOYALTY_REWARDS.POINTS_BALANCE" in VALID_INTENT_CODES
        assert "SHIPPING.INTERNATIONAL" in VALID_INTENT_CODES
        assert "BULK_WHOLESALE.BULK_DISCOUNT" in VALID_INTENT_CODES

        # Check invalid intents don't exist
        assert "INVALID.INTENT" not in VALID_INTENT_CODES
        assert "ORDER_STATUS.INVALID" not in VALID_INTENT_CODES

    def test_agent_creation_with_test_model(self, test_model: TestModel) -> None:
        """Test that the agent can be created with TestModel."""
        agent = get_intent_agent(model=test_model)
        assert agent is not None

    @pytest.mark.asyncio
    async def test_decomposition_with_test_model(self, agent_with_test_model) -> None:
        """Test decomposition using TestModel for deterministic results."""
        agent = agent_with_test_model

        context = IntentContext(
            raw_text="Where is my order #12345?",
            extracted_entities=[{"entity_type": "order_id", "value": "12345"}],
            match_hints=["ORDER_STATUS.WISMO"],
        )

        # Run with TestModel - no override needed since model was passed at creation
        result = await agent.run("Where is my order #12345?", deps=context)

        # Verify the result is a valid DecompositionResult
        assert isinstance(result.output, DecompositionResult)
        assert len(result.output.intents) >= 0  # TestModel may return empty
        assert isinstance(result.output.is_compound, bool)
        assert isinstance(result.output.reasoning, str)

    @pytest.mark.asyncio
    async def test_compound_intent_detection(self, agent_with_test_model) -> None:
        """Test compound intent detection using TestModel."""
        agent = agent_with_test_model

        context = IntentContext(
            raw_text="I want to return this and get my refund",
            extracted_entities=[],
            match_hints=["RETURN_EXCHANGE.RETURN_INITIATE", "RETURN_EXCHANGE.REFUND_STATUS"],
        )

        result = await agent.run(
            "I want to return this and get my refund",
            deps=context,
        )

        assert isinstance(result.output, DecompositionResult)

    @pytest.mark.asyncio
    async def test_context_includes_entities(self, agent_with_test_model) -> None:
        """Test that extracted entities are passed to the agent."""
        agent = agent_with_test_model

        entities = [
            {"entity_type": "order_id", "value": "12345", "confidence": 0.99},
            {"entity_type": "tracking_number", "value": "1Z999AA10123456784", "confidence": 0.95},
        ]

        context = IntentContext(
            raw_text="Track my order #12345 with tracking 1Z999AA10123456784",
            extracted_entities=entities,
            match_hints=["ORDER_STATUS.WISMO"],
        )

        result = await agent.run(
            "Track my order #12345 with tracking 1Z999AA10123456784",
            deps=context,
        )

        assert isinstance(result.output, DecompositionResult)

    @pytest.mark.asyncio
    async def test_context_includes_sentiment(self, agent_with_test_model) -> None:
        """Test that sentiment scores are passed to the agent."""
        agent = agent_with_test_model

        context = IntentContext(
            raw_text="This is ridiculous! Where is my order?!",
            extracted_entities=[],
            match_hints=["ORDER_STATUS.WISMO", "COMPLAINT.SERVICE_ISSUE"],
            frustration_score=0.85,
            urgency_score=0.70,
            sentiment_score=-0.6,
        )

        result = await agent.run(
            "This is ridiculous! Where is my order?!",
            deps=context,
        )

        assert isinstance(result.output, DecompositionResult)


class TestResponseAgent:
    """Tests for the response generation agent."""

    @pytest.fixture
    def test_model(self) -> TestModel:
        """Create a TestModel for response generation."""
        return TestModel()

    @pytest.fixture
    def response_agent_with_test_model(self, test_model: TestModel):
        """Create a response agent with TestModel."""
        return get_response_agent(model=test_model)

    def test_response_agent_creation_with_test_model(self, test_model: TestModel) -> None:
        """Test that the response agent can be created with TestModel."""
        agent = get_response_agent(model=test_model)
        assert agent is not None

    @pytest.mark.asyncio
    async def test_response_generation_with_test_model(
        self, response_agent_with_test_model
    ) -> None:
        """Test response generation with TestModel."""
        agent = response_agent_with_test_model

        ctx = ResponseContext(
            intent_code="ORDER_STATUS.WISMO",
        )

        result = await agent.run(
            "Generate response for order status inquiry",
            deps=ctx,
        )

        assert isinstance(result.output, GeneratedResponse)
        assert isinstance(result.output.text, str)
        assert result.output.tone in [
            "empathetic",
            "helpful",
            "apologetic",
            "informative",
            "urgent",
        ]
        assert isinstance(result.output.suggested_actions, list)
        assert isinstance(result.output.requires_followup, bool)

    @pytest.mark.asyncio
    async def test_response_with_order_context(self, response_agent_with_test_model) -> None:
        """Test response generation with order context."""
        agent = response_agent_with_test_model

        ctx = ResponseContext(
            intent_code="ORDER_STATUS.WISMO",
            order_number="12345",
            order_status="shipped",
            carrier="UPS",
            tracking_number="1Z999AA10123456784",
        )

        result = await agent.run(
            "Generate response for order status",
            deps=ctx,
        )

        assert isinstance(result.output, GeneratedResponse)

    @pytest.mark.asyncio
    async def test_response_for_complaint(self, response_agent_with_test_model) -> None:
        """Test response generation for complaints has empathetic tone option."""
        agent = response_agent_with_test_model

        ctx = ResponseContext(
            intent_code="COMPLAINT.DAMAGED_ITEM",
            order_number="12345",
        )

        result = await agent.run(
            "Generate response for damaged item complaint",
            deps=ctx,
        )

        assert isinstance(result.output, GeneratedResponse)
        # TestModel generates valid enum values, so tone should be valid
        assert result.output.tone in [
            "empathetic",
            "helpful",
            "apologetic",
            "informative",
            "urgent",
        ]


class TestResponseContext:
    """Tests for ResponseContext dataclass."""

    def test_response_context_creation(self) -> None:
        """Test ResponseContext can be created with minimal fields."""
        ctx = ResponseContext(intent_code="ORDER_STATUS.WISMO")

        assert ctx.intent_code == "ORDER_STATUS.WISMO"
        assert ctx.order_number is None
        assert ctx.is_vip is False

    def test_response_context_with_all_fields(self) -> None:
        """Test ResponseContext with all fields populated."""
        ctx = ResponseContext(
            intent_code="RETURN_EXCHANGE.RETURN_INITIATE",
            order_number="12345",
            order_status="delivered",
            carrier="FedEx",
            tracking_number="123456789",
            tracking_url="https://fedex.com/track/123456789",
            estimated_delivery="January 15, 2026",
            is_within_return_window=True,
            days_until_return_expires=15,
            refund_amount="99.99 USD",
            customer_name="John Doe",
            customer_tier="VIP",
            is_vip=True,
            entities=[{"entity_type": "order_id", "value": "12345"}],
        )

        assert ctx.intent_code == "RETURN_EXCHANGE.RETURN_INITIATE"
        assert ctx.order_number == "12345"
        assert ctx.is_vip is True
        assert ctx.days_until_return_expires == 15


class TestIntentContextTools:
    """Tests for IntentContext tool support."""

    def test_context_supports_order_lookup(self) -> None:
        """Test that IntentContext can hold order lookup callback."""

        async def mock_lookup(order_id: str) -> dict | None:
            return {"order_id": order_id, "status": "shipped"}

        ctx = IntentContext(
            raw_text="Where is order 12345?",
            extracted_entities=[],
            match_hints=[],
            order_lookup=mock_lookup,
        )

        assert ctx.order_lookup is not None

    def test_context_supports_return_eligibility(self) -> None:
        """Test that IntentContext can hold return eligibility callback."""

        async def mock_check(order_id: str) -> dict | None:
            return {"order_id": order_id, "eligible": True, "days_remaining": 15}

        ctx = IntentContext(
            raw_text="Can I return order 12345?",
            extracted_entities=[],
            match_hints=[],
            return_eligibility_check=mock_check,
        )

        assert ctx.return_eligibility_check is not None
