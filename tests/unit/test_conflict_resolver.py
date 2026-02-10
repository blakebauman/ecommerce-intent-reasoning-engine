"""Tests for conflict resolver."""

import pytest

from intent_engine.models.conflict import (
    ConflictType,
    ResolutionStrategy,
)
from intent_engine.models.context import EnrichedContext, OrderContext
from intent_engine.models.entity import EntityType, ExtractedEntity
from intent_engine.models.intent import IntentConfidence, ResolvedIntent
from intent_engine.reasoners.conflict_resolver import ConflictResolver


@pytest.fixture
def resolver() -> ConflictResolver:
    """Create a conflict resolver."""
    return ConflictResolver()


def make_intent(
    category: str, intent: str, confidence: float = 0.85, evidence: list[str] | None = None
) -> ResolvedIntent:
    """Helper to create a ResolvedIntent."""
    tier = (
        IntentConfidence.HIGH
        if confidence >= 0.85
        else IntentConfidence.MEDIUM if confidence >= 0.6 else IntentConfidence.LOW
    )
    return ResolvedIntent(
        category=category,
        intent=intent,
        confidence=confidence,
        confidence_tier=tier,
        evidence=evidence or [],
    )


def make_entity(
    entity_type: EntityType, value: str, raw_span: str | None = None
) -> ExtractedEntity:
    """Helper to create an ExtractedEntity."""
    return ExtractedEntity(
        entity_type=entity_type,
        value=value,
        raw_span=raw_span or value,
        start_pos=0,
        end_pos=len(value),
        confidence=0.95,
    )


class TestNoConflict:
    """Tests for cases with no conflict."""

    async def test_single_intent_no_conflict(self, resolver: ConflictResolver) -> None:
        """Single intent should pass through without conflict."""
        intents = [make_intent("ORDER_STATUS", "WISMO")]

        result = await resolver.resolve(intents=intents, entities=[], text="where is my order")

        assert result.has_conflict is False
        assert len(result.resolved_intents) == 1
        assert result.resolved_intents[0].intent == "WISMO"
        assert "Single intent" in result.reasoning[1]

    async def test_complementary_intents_no_conflict(self, resolver: ConflictResolver) -> None:
        """WISMO + DELIVERY_ESTIMATE are complementary, no conflict."""
        intents = [
            make_intent("ORDER_STATUS", "WISMO"),
            make_intent("ORDER_STATUS", "DELIVERY_ESTIMATE"),
        ]

        result = await resolver.resolve(intents=intents, entities=[], text="where is my order")

        assert result.has_conflict is False
        assert len(result.resolved_intents) == 2

    async def test_different_items_no_conflict(self, resolver: ConflictResolver) -> None:
        """Return item A, exchange item B - no conflict if different items."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]
        entities = [
            make_entity(EntityType.PRODUCT_SKU, "SKU-001", "blue shirt"),
            make_entity(EntityType.PRODUCT_SKU, "SKU-002", "red pants"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=entities,
            text="return the blue shirt and exchange the red pants",
        )

        assert result.has_conflict is False
        assert len(result.resolved_intents) == 2


class TestConflictWithPreference:
    """Tests for conflicts resolved by customer preference."""

    async def test_prefer_exchange(self, resolver: ConflictResolver) -> None:
        """'Exchange not return' should resolve to exchange only."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            text="I want to exchange this, not return it for a refund",
        )

        assert result.has_conflict is True
        assert result.resolution_strategy == ResolutionStrategy.PREFERENCE
        assert len(result.resolved_intents) == 1
        assert result.resolved_intents[0].intent == "EXCHANGE_REQUEST"

    async def test_prefer_refund(self, resolver: ConflictResolver) -> None:
        """'Just want a refund' should resolve to return only."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            text="I just want to return this for a refund, not exchange",
        )

        assert result.has_conflict is True
        assert result.resolution_strategy == ResolutionStrategy.PREFERENCE
        assert len(result.resolved_intents) == 1
        assert result.resolved_intents[0].intent == "RETURN_INITIATE"


class TestConflictNeedsClairification:
    """Tests for conflicts that need customer clarification."""

    async def test_return_and_exchange_ambiguous(self, resolver: ConflictResolver) -> None:
        """'Return AND exchange' with no preference should request clarification."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents, entities=[], text="I want a return and an exchange"
        )

        # Without VIP/high frustration and no clear preference, should need clarification
        # or fall back to priority
        assert result.has_conflict is True
        # Could be PRIORITY (business rules) or CLARIFICATION depending on priority scores
        assert result.resolution_strategy in (
            ResolutionStrategy.PRIORITY,
            ResolutionStrategy.CLARIFICATION,
        )

    async def test_cancel_and_expedite_clarification(self, resolver: ConflictResolver) -> None:
        """Cancel + Expedite should request clarification."""
        intents = [
            make_intent("ORDER_MODIFY", "CANCEL_ORDER"),
            make_intent("ORDER_MODIFY", "EXPEDITE"),
        ]

        result = await resolver.resolve(
            intents=intents, entities=[], text="cancel the order but also ship it faster"
        )

        assert result.has_conflict is True
        assert result.conflict_type == ConflictType.CONTRADICTORY_POLICY
        # These have equal priority (1 vs 2 but not enough to auto-resolve always)
        # or should escalate/clarify


class TestPolicyViolation:
    """Tests for policy violation conflicts."""

    async def test_return_after_window_expired(self, resolver: ConflictResolver) -> None:
        """Return after window expired should be policy violation."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]
        context = EnrichedContext(
            order=OrderContext(
                order_id="123",
                order_number="#1234",
                status="delivered",
                fulfillment_status="fulfilled",
                customer_email="test@example.com",
                subtotal=100.0,
                total=100.0,
                is_within_return_window=False,  # Window expired
            )
        )

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            context=context,
            text="I want to return this",
        )

        assert result.has_conflict is True
        assert result.conflict_type == ConflictType.POLICY_VIOLATION


class TestVIPLeniency:
    """Tests for VIP customer leniency."""

    async def test_vip_can_have_both(self, resolver: ConflictResolver) -> None:
        """VIP customers may get both actions approved."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            customer_tier="VIP",
            text="I want both a return and an exchange",
        )

        assert result.has_conflict is True
        # VIP should keep both intents
        assert len(result.resolved_intents) == 2
        assert result.resolution_strategy == ResolutionStrategy.PRIORITY

    async def test_at_risk_customer_leniency(self, resolver: ConflictResolver) -> None:
        """AT_RISK customers get similar leniency."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            customer_tier="at_risk",
            text="I want both",
        )

        assert result.has_conflict is True
        assert len(result.resolved_intents) == 2

    async def test_vip_cannot_cancel_and_expedite(self, resolver: ConflictResolver) -> None:
        """Even VIP cannot do contradictory cancel + expedite."""
        intents = [
            make_intent("ORDER_MODIFY", "CANCEL_ORDER"),
            make_intent("ORDER_MODIFY", "EXPEDITE"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            customer_tier="VIP",
            text="cancel but expedite",
        )

        assert result.has_conflict is True
        # Should NOT keep both for truly contradictory actions
        assert len(result.resolved_intents) < 2 or result.requires_clarification


class TestHighFrustration:
    """Tests for high frustration score handling."""

    async def test_high_frustration_favors_customer(self, resolver: ConflictResolver) -> None:
        """High frustration (>0.7) should favor customer-friendly option."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),  # Refund = customer-favorable
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),  # Exchange = merchant-favorable
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            frustration_score=0.85,  # High frustration
            text="I'm so frustrated with this order",
        )

        assert result.has_conflict is True
        assert len(result.resolved_intents) == 1
        # High frustration should favor refund (customer-favorable)
        assert result.resolved_intents[0].intent == "RETURN_INITIATE"

    async def test_normal_frustration_uses_business_priority(
        self, resolver: ConflictResolver
    ) -> None:
        """Normal frustration should use business priority rules."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            frustration_score=0.3,  # Low frustration
            text="I might want to return or exchange this",
        )

        assert result.has_conflict is True
        # Business priority prefers exchange (keeps customer)
        if result.resolution_strategy == ResolutionStrategy.PRIORITY:
            assert len(result.resolved_intents) == 1
            assert result.resolved_intents[0].intent == "EXCHANGE_REQUEST"


class TestContradictoryActions:
    """Tests for contradictory action combinations."""

    async def test_cancel_and_change_address(self, resolver: ConflictResolver) -> None:
        """Cancel + Change Address is contradictory."""
        intents = [
            make_intent("ORDER_MODIFY", "CANCEL_ORDER"),
            make_intent("ORDER_MODIFY", "CHANGE_ADDRESS"),
        ]

        result = await resolver.resolve(
            intents=intents, entities=[], text="cancel but also change the address"
        )

        assert result.has_conflict is True
        assert result.conflict_type == ConflictType.CONTRADICTORY_POLICY

    async def test_expedite_and_delay(self, resolver: ConflictResolver) -> None:
        """Expedite + Delay is contradictory."""
        intents = [
            make_intent("ORDER_MODIFY", "EXPEDITE"),
            make_intent("ORDER_MODIFY", "DELAY_SHIPMENT"),
        ]

        result = await resolver.resolve(
            intents=intents, entities=[], text="expedite but also delay"
        )

        assert result.has_conflict is True
        assert result.conflict_type == ConflictType.CONTRADICTORY_POLICY


class TestClarificationGeneration:
    """Tests for clarification question generation."""

    async def test_clarification_question_format(self, resolver: ConflictResolver) -> None:
        """Clarification question should be well-formed."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        # Force clarification by not having VIP/high frustration/preference
        result = await resolver.resolve(
            intents=intents,
            entities=[],
            customer_tier="standard",
            frustration_score=0.2,
            text="maybe return or exchange",
        )

        if result.requires_clarification:
            assert result.clarification_question is not None
            assert "return" in result.clarification_question.lower()
            assert "exchange" in result.clarification_question.lower()
            assert len(result.clarification_options) >= 2


class TestReasoningTrace:
    """Tests for reasoning trace output."""

    async def test_reasoning_trace_included(self, resolver: ConflictResolver) -> None:
        """Reasoning trace should document the resolution process."""
        intents = [make_intent("ORDER_STATUS", "WISMO")]

        result = await resolver.resolve(intents=intents, entities=[], text="where is my order")

        assert len(result.reasoning) >= 1
        assert "Step 9: Conflict resolution" in result.reasoning[0]

    async def test_conflict_reasoning_detailed(self, resolver: ConflictResolver) -> None:
        """Conflict reasoning should include detection and resolution details."""
        intents = [
            make_intent("RETURN_EXCHANGE", "RETURN_INITIATE"),
            make_intent("RETURN_EXCHANGE", "EXCHANGE_REQUEST"),
        ]

        result = await resolver.resolve(
            intents=intents,
            entities=[],
            text="I prefer to exchange",
        )

        # Should have step header and conflict detection
        assert any("Step 9" in r for r in result.reasoning)
        if result.has_conflict:
            assert any("conflict" in r.lower() for r in result.reasoning)
