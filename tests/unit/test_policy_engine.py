"""Tests for policy engine."""

import pytest
from datetime import datetime, timedelta

from intent_engine.reasoners.policy_engine import PolicyEngine, PolicyDecision
from intent_engine.models.context import (
    CustomerProfile,
    CustomerTier,
    OrderContext,
    ProductContext,
    EnrichedContext,
    ReturnEligibility,
)


@pytest.fixture
def default_policy() -> dict:
    """Create a default test policy."""
    return {
        "tenant_id": "default",
        "version": "1.0.0",
        "return_policy": {
            "default_window_days": 30,
            "premium_window_days": 45,
            "vip_window_days": 60,
            "final_sale_categories": ["clearance", "swimwear", "undergarments"],
        },
        "auto_approval": {
            "enabled": True,
            "return": {
                "max_amount_standard": 100,
                "max_amount_premium": 200,
                "max_amount_vip": 500,
                "excluded_categories": ["electronics", "luxury"],
            },
            "refund": {
                "max_amount_standard": 50,
                "max_amount_premium": 100,
                "max_amount_vip": 250,
            },
            "replacement": {
                "enabled": True,
                "max_amount": 200,
            },
        },
        "escalation": {
            "complaint_threshold": 3,
            "high_value_threshold": 500,
            "frustration_score_threshold": 0.7,
            "auto_escalate_keywords": ["lawyer", "sue", "legal action"],
        },
        "priority_routing": {
            "enabled": True,
            "vip_priority": True,
            "high_frustration_priority": True,
            "frustration_threshold": 0.7,
            "high_value_order_threshold": 300,
        },
    }


@pytest.fixture
def engine(default_policy: dict) -> PolicyEngine:
    """Create a policy engine with default policy."""
    return PolicyEngine(default_policy=default_policy)


@pytest.fixture
def standard_customer() -> CustomerProfile:
    """Create a standard tier customer."""
    return CustomerProfile(
        customer_id="cust-123",
        email="customer@example.com",
        tier=CustomerTier.STANDARD,
        lifetime_value=500.0,
        total_orders=5,
        complaints_90d=0,
    )


@pytest.fixture
def vip_customer() -> CustomerProfile:
    """Create a VIP tier customer."""
    return CustomerProfile(
        customer_id="cust-456",
        email="vip@example.com",
        tier=CustomerTier.VIP,
        lifetime_value=10000.0,
        total_orders=50,
        complaints_90d=0,
        is_vip=True,
    )


@pytest.fixture
def recent_order() -> OrderContext:
    """Create a recent order within return window."""
    return OrderContext(
        order_id="ORD-12345",
        order_number="12345",
        total=75.00,
        subtotal=75.00,
        status="delivered",
        fulfillment_status="fulfilled",
        customer_email="customer@example.com",
        created_at=datetime.utcnow() - timedelta(days=10),
        items=[
            ProductContext(
                product_id="PROD-001",
                sku="SKU-001",
                name="T-Shirt",
                price=25.00,
                category="apparel",
            )
        ],
        is_within_return_window=True,
        days_until_return_expires=20,
    )


@pytest.fixture
def expired_order() -> OrderContext:
    """Create an order with expired return window."""
    return OrderContext(
        order_id="ORD-OLD",
        order_number="OLD123",
        total=150.00,
        subtotal=150.00,
        status="delivered",
        fulfillment_status="fulfilled",
        customer_email="customer@example.com",
        created_at=datetime.utcnow() - timedelta(days=60),
        items=[ProductContext(product_id="PROD-002", sku="SKU-002", name="Pants", price=150.00)],
        is_within_return_window=False,
        return_eligibility=ReturnEligibility.EXPIRED,
    )


class TestPolicyEngineBasic:
    """Basic policy engine tests."""

    def test_initialization(self, engine: PolicyEngine) -> None:
        """Test engine initializes correctly."""
        assert engine is not None
        policy = engine.get_policy("default")
        assert policy["version"] == "1.0.0"

    def test_get_policy_fallback(self, engine: PolicyEngine) -> None:
        """Test fallback to default policy."""
        policy = engine.get_policy("unknown-tenant")
        assert policy["tenant_id"] == "default"

    def test_parse_intent_code(self, engine: PolicyEngine) -> None:
        """Test intent code parsing."""
        category, intent = engine._parse_intent_code("RETURN_EXCHANGE.RETURN_INITIATE")
        assert category == "RETURN_EXCHANGE"
        assert intent == "RETURN_INITIATE"

    def test_parse_intent_code_single(self, engine: PolicyEngine) -> None:
        """Test single-part intent code."""
        category, intent = engine._parse_intent_code("WISMO")
        assert category == "WISMO"
        assert intent == ""


class TestReturnEligibility:
    """Tests for return window validation."""

    def test_eligible_return_recent_order(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test return is eligible for recent order."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.return_eligible is True
        assert decision.days_until_return_expires == 20

    def test_ineligible_expired_window(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        expired_order: OrderContext,
    ) -> None:
        """Test return is ineligible when window expired."""
        context = EnrichedContext(customer=standard_customer, order=expired_order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.return_eligible is False
        assert "expired" in decision.return_ineligible_reason.lower()

    def test_ineligible_final_sale(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test return is ineligible for final sale items."""
        order = OrderContext(
            order_id="ORD-FINAL",
            order_number="FINAL",
            total=50.00,
            subtotal=50.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
            items=[
                ProductContext(
                    product_id="PROD-SWIM",
                    sku="SKU-SWIM",
                    name="Swimsuit",
                    price=50.00,
                    category="swimwear",
                )
            ],
            is_within_return_window=True,
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.return_eligible is False
        assert "final sale" in decision.return_ineligible_reason.lower()

    def test_ineligible_cancelled_order(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test return is ineligible for cancelled orders."""
        order = OrderContext(
            order_id="ORD-CANCEL",
            order_number="CANCEL",
            total=100.00,
            subtotal=100.00,
            status="cancelled",
            fulfillment_status="unfulfilled",
            customer_email="customer@example.com",
            is_cancelled=True,
            is_within_return_window=True,
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.return_eligible is False
        assert "cancelled" in decision.return_ineligible_reason.lower()


class TestAutoApproval:
    """Tests for auto-approval logic."""

    def test_auto_approve_return_under_threshold(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test return auto-approved under threshold."""
        # Order total is 75, threshold for standard is 100
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.auto_approve_return is True
        assert decision.recommended_action == "auto_approve_return"

    def test_no_auto_approve_over_threshold(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test return not auto-approved over threshold."""
        order = OrderContext(
            order_id="ORD-BIG",
            order_number="BIG",
            total=150.00,  # Over 100 threshold
            subtotal=150.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
            items=[ProductContext(product_id="PROD-001", sku="SKU-001", name="Item", price=150.00)],
            is_within_return_window=True,
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.auto_approve_return is False

    def test_vip_higher_threshold(
        self, engine: PolicyEngine, vip_customer: CustomerProfile
    ) -> None:
        """Test VIP has higher auto-approval threshold."""
        order = OrderContext(
            order_id="ORD-VIP",
            order_number="VIP",
            total=400.00,  # Under VIP's 500 threshold
            subtotal=400.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="vip@example.com",
            items=[ProductContext(product_id="PROD-001", sku="SKU-001", name="Item", price=400.00)],
            is_within_return_window=True,
        )

        context = EnrichedContext(customer=vip_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.auto_approve_return is True

    def test_no_auto_approve_excluded_category(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test no auto-approval for excluded categories."""
        order = OrderContext(
            order_id="ORD-ELEC",
            order_number="ELEC",
            total=50.00,  # Under threshold
            subtotal=50.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
            items=[
                ProductContext(
                    product_id="PROD-ELEC",
                    sku="SKU-ELEC",
                    name="Headphones",
                    price=50.00,
                    category="electronics",
                )
            ],
            is_within_return_window=True,
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.auto_approve_return is False

    def test_auto_approve_refund(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test refund auto-approval."""
        order = OrderContext(
            order_id="ORD-REF",
            order_number="REF",
            total=40.00,  # Under 50 threshold
            subtotal=40.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.REFUND_REQUEST")

        assert decision.auto_approve_refund is True

    def test_auto_approve_replacement(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test replacement auto-approval."""
        order = OrderContext(
            order_id="ORD-REPL",
            order_number="REPL",
            total=150.00,
            subtotal=150.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.EXCHANGE_REQUEST")

        assert decision.auto_approve_replacement is True


class TestEscalation:
    """Tests for escalation triggers."""

    def test_escalate_high_complaints(
        self, engine: PolicyEngine, recent_order: OrderContext
    ) -> None:
        """Test escalation for customers with many complaints."""
        customer = CustomerProfile(
            customer_id="cust-problem",
            email="problem@example.com",
            tier=CustomerTier.STANDARD,
            complaints_90d=5,  # Over threshold of 3
        )

        context = EnrichedContext(customer=customer, order=recent_order)
        decision = engine.evaluate(context, "COMPLAINT.GENERAL")

        assert decision.escalation_required is True
        assert any("complaints" in r.lower() for r in decision.escalation_reasons)

    def test_escalate_high_value_order(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test escalation for high-value orders."""
        order = OrderContext(
            order_id="ORD-HIGHVAL",
            order_number="HIGHVAL",
            total=600.00,  # Over 500 threshold
            subtotal=600.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "COMPLAINT.GENERAL")

        assert decision.escalation_required is True
        assert any("high-value" in r.lower() for r in decision.escalation_reasons)

    def test_escalate_high_frustration(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test escalation for high frustration score."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(
            context,
            "COMPLAINT.GENERAL",
            frustration_score=0.85,  # Over 0.7 threshold
        )

        assert decision.escalation_required is True
        assert any("frustration" in r.lower() for r in decision.escalation_reasons)

    def test_no_escalation_normal_case(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test no escalation for normal cases."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "ORDER_STATUS.WISMO")

        assert decision.escalation_required is False

    def test_check_escalation_keywords(
        self, engine: PolicyEngine, default_policy: dict
    ) -> None:
        """Test keyword-based escalation checking."""
        text = "I will sue you! My lawyer will be in touch!"
        keywords = engine.check_escalation_keywords(text, default_policy)

        assert "lawyer" in keywords
        assert "sue" in keywords


class TestPriorityRouting:
    """Tests for priority routing."""

    def test_vip_priority(
        self, engine: PolicyEngine, vip_customer: CustomerProfile
    ) -> None:
        """Test VIP customers get priority."""
        context = EnrichedContext(customer=vip_customer)
        decision = engine.evaluate(context, "ORDER_STATUS.WISMO")

        assert decision.priority_flag is True
        assert any("vip" in r.lower() for r in decision.priority_reasons)

    def test_frustration_priority(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test high frustration gets priority."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(
            context, "ORDER_STATUS.WISMO", frustration_score=0.8
        )

        assert decision.priority_flag is True
        assert any("frustration" in r.lower() for r in decision.priority_reasons)

    def test_high_value_order_priority(
        self, engine: PolicyEngine, standard_customer: CustomerProfile
    ) -> None:
        """Test high-value orders get priority."""
        order = OrderContext(
            order_id="ORD-PRIO",
            order_number="PRIO",
            total=350.00,  # Over 300 threshold
            subtotal=350.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
        )

        context = EnrichedContext(customer=standard_customer, order=order)
        decision = engine.evaluate(context, "ORDER_STATUS.WISMO")

        assert decision.priority_flag is True
        assert any("high-value" in r.lower() for r in decision.priority_reasons)

    def test_no_priority_normal_case(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test no priority for normal cases."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "ORDER_STATUS.WISMO", frustration_score=0.2)

        assert decision.priority_flag is False


class TestRecommendations:
    """Tests for action recommendations."""

    def test_recommend_auto_return(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test return recommendation when auto-approved."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.recommended_action == "auto_approve_return"
        assert "return label" in decision.suggested_resolution.lower()

    def test_recommend_escalation(
        self, engine: PolicyEngine, recent_order: OrderContext
    ) -> None:
        """Test escalation recommendation."""
        customer = CustomerProfile(
            customer_id="cust-esc",
            email="escalate@example.com",
            tier=CustomerTier.STANDARD,
            complaints_90d=5,
        )

        context = EnrichedContext(customer=customer, order=recent_order)
        decision = engine.evaluate(context, "COMPLAINT.GENERAL")

        assert decision.recommended_action == "escalate_to_supervisor"

    def test_recommend_explain_policy(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        expired_order: OrderContext,
    ) -> None:
        """Test policy explanation recommendation for ineligible returns."""
        context = EnrichedContext(customer=standard_customer, order=expired_order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert decision.recommended_action == "explain_policy"


class TestReturnWindowValidation:
    """Tests for return window validation helper."""

    def test_validate_standard_window(
        self, engine: PolicyEngine, recent_order: OrderContext
    ) -> None:
        """Test standard 30-day window."""
        is_eligible, reason, days = engine.validate_return_window(
            recent_order, CustomerTier.STANDARD
        )

        assert is_eligible is True
        assert reason is None
        assert days is not None and days > 0

    def test_validate_vip_extended_window(
        self, engine: PolicyEngine
    ) -> None:
        """Test VIP extended 60-day window."""
        order = OrderContext(
            order_id="ORD-VIP-WIN",
            order_number="VIP-WIN",
            total=100.00,
            subtotal=100.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
            created_at=datetime.utcnow() - timedelta(days=45),  # 45 days old
        )

        # Standard customer would be outside window (45 > 30)
        std_eligible, std_reason, _ = engine.validate_return_window(
            order, CustomerTier.STANDARD
        )

        # VIP customer should be within window (45 < 60)
        vip_eligible, vip_reason, _ = engine.validate_return_window(
            order, CustomerTier.VIP
        )

        assert std_eligible is False
        assert vip_eligible is True

    def test_validate_expired_window(self, engine: PolicyEngine) -> None:
        """Test expired window."""
        order = OrderContext(
            order_id="ORD-EXP",
            order_number="EXP",
            total=100.00,
            subtotal=100.00,
            status="delivered",
            fulfillment_status="fulfilled",
            customer_email="customer@example.com",
            created_at=datetime.utcnow() - timedelta(days=100),
        )

        is_eligible, reason, days = engine.validate_return_window(
            order, CustomerTier.STANDARD
        )

        assert is_eligible is False
        assert "expired" in reason.lower()
        assert days is not None and days < 0


class TestPolicyDecision:
    """Tests for PolicyDecision structure."""

    def test_decision_defaults(self) -> None:
        """Test PolicyDecision has correct defaults."""
        decision = PolicyDecision()

        assert decision.auto_approve_return is False
        assert decision.auto_approve_refund is False
        assert decision.escalation_required is False
        assert decision.priority_flag is False
        assert decision.return_eligible is True
        assert decision.rules_applied == []

    def test_decision_includes_timestamp(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test decision includes evaluation timestamp."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "ORDER_STATUS.WISMO")

        assert decision.evaluated_at is not None
        assert isinstance(decision.evaluated_at, datetime)

    def test_decision_tracks_rules(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test decision tracks which rules were applied."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)
        decision = engine.evaluate(context, "RETURN_EXCHANGE.RETURN_INITIATE")

        assert "return_eligibility" in decision.rules_applied
        assert "auto_approval" in decision.rules_applied
        assert "escalation" in decision.rules_applied
        assert "priority_routing" in decision.rules_applied


class TestTierAwareFrustrationThresholds:
    """Tests for tier-aware frustration thresholds (Phase 2)."""

    @pytest.fixture
    def at_risk_customer(self) -> CustomerProfile:
        """Create an at-risk tier customer."""
        return CustomerProfile(
            customer_id="cust-atrisk",
            email="atrisk@example.com",
            tier=CustomerTier.AT_RISK,
            lifetime_value=100.0,
            total_orders=2,
            complaints_90d=2,
        )

    def test_vip_higher_frustration_threshold(
        self,
        engine: PolicyEngine,
        vip_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test VIP customers have higher frustration threshold (0.8)."""
        context = EnrichedContext(customer=vip_customer, order=recent_order)

        # 0.75 should NOT escalate for VIP (threshold is 0.8)
        decision = engine.evaluate(
            context, "COMPLAINT.GENERAL", frustration_score=0.75
        )
        frustration_reasons = [r for r in decision.escalation_reasons if "frustration" in r.lower()]
        assert len(frustration_reasons) == 0, "VIP should not escalate at 0.75 frustration"

        # 0.85 SHOULD escalate for VIP
        decision_high = engine.evaluate(
            context, "COMPLAINT.GENERAL", frustration_score=0.85
        )
        frustration_reasons_high = [r for r in decision_high.escalation_reasons if "frustration" in r.lower()]
        assert len(frustration_reasons_high) > 0, "VIP should escalate at 0.85 frustration"

    def test_at_risk_lower_frustration_threshold(
        self,
        engine: PolicyEngine,
        at_risk_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test at-risk customers have lower frustration threshold (0.5)."""
        context = EnrichedContext(customer=at_risk_customer, order=recent_order)

        # 0.55 SHOULD escalate for AT_RISK (threshold is 0.5)
        decision = engine.evaluate(
            context, "COMPLAINT.GENERAL", frustration_score=0.55
        )
        frustration_reasons = [r for r in decision.escalation_reasons if "frustration" in r.lower()]
        assert len(frustration_reasons) > 0, "AT_RISK should escalate at 0.55 frustration"

    def test_standard_customer_default_threshold(
        self,
        engine: PolicyEngine,
        standard_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test standard customers use default threshold (0.7)."""
        context = EnrichedContext(customer=standard_customer, order=recent_order)

        # 0.65 should NOT escalate for standard
        decision = engine.evaluate(
            context, "COMPLAINT.GENERAL", frustration_score=0.65
        )
        frustration_reasons = [r for r in decision.escalation_reasons if "frustration" in r.lower()]
        assert len(frustration_reasons) == 0, "Standard should not escalate at 0.65 frustration"

        # 0.75 SHOULD escalate for standard
        decision_high = engine.evaluate(
            context, "COMPLAINT.GENERAL", frustration_score=0.75
        )
        frustration_reasons_high = [r for r in decision_high.escalation_reasons if "frustration" in r.lower()]
        assert len(frustration_reasons_high) > 0, "Standard should escalate at 0.75 frustration"

    def test_tier_shown_in_escalation_reason(
        self,
        engine: PolicyEngine,
        vip_customer: CustomerProfile,
        recent_order: OrderContext,
    ) -> None:
        """Test that tier is shown in escalation reason."""
        context = EnrichedContext(customer=vip_customer, order=recent_order)
        decision = engine.evaluate(
            context, "COMPLAINT.GENERAL", frustration_score=0.9
        )

        # Find the frustration-related escalation reason
        frustration_reasons = [r for r in decision.escalation_reasons if "frustration" in r.lower()]
        assert len(frustration_reasons) > 0
        # The reason should mention the tier
        assert "vip" in frustration_reasons[0].lower()
