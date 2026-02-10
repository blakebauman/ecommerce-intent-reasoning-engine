"""Policy engine for business rule evaluation."""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from intent_engine.models.context import (
    CustomerProfile,
    CustomerTier,
    EnrichedContext,
    OrderContext,
    PolicyContext,
    ReturnEligibility,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """Result of policy evaluation."""

    # Approval decisions
    auto_approve_return: bool = False
    auto_approve_refund: bool = False
    auto_approve_replacement: bool = False

    # Escalation
    escalation_required: bool = False
    escalation_reasons: list[str] = field(default_factory=list)

    # Priority routing
    priority_flag: bool = False
    priority_reasons: list[str] = field(default_factory=list)

    # Constraints
    return_eligible: bool = True
    return_ineligible_reason: str | None = None
    days_until_return_expires: int | None = None

    # Recommendations
    recommended_action: str | None = None
    suggested_resolution: str | None = None

    # Metadata
    policy_version: str = "1.0.0"
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    rules_applied: list[str] = field(default_factory=list)


class PolicyEngine:
    """
    Evaluate business rules and policies for intent resolution.

    Handles:
    - Return window validation
    - Auto-approval thresholds (per tier, per amount)
    - Escalation triggers (3+ complaints → escalate)
    - Priority routing based on frustration/VIP status
    - Per-tenant policy configuration
    """

    DEFAULT_POLICY_PATH = Path(__file__).parent.parent.parent.parent / "data" / "policies"

    def __init__(
        self,
        policy_path: Path | str | None = None,
        default_policy: dict | None = None,
    ) -> None:
        """
        Initialize the policy engine.

        Args:
            policy_path: Path to policy configuration files.
            default_policy: Default policy dict (overrides file loading).
        """
        self.policy_path = Path(policy_path) if policy_path else self.DEFAULT_POLICY_PATH
        self._policies: dict[str, dict] = {}
        self._default_policy = default_policy

        # Load default policy
        if default_policy:
            self._policies["default"] = default_policy
        else:
            self._load_policies()

    def _load_policies(self) -> None:
        """Load policy configurations from files."""
        if not self.policy_path.exists():
            logger.warning(f"Policy path does not exist: {self.policy_path}")
            return

        for policy_file in self.policy_path.glob("*.json"):
            try:
                with open(policy_file) as f:
                    policy = json.load(f)
                    tenant_id = policy.get("tenant_id", policy_file.stem)
                    self._policies[tenant_id] = policy
                    logger.info(f"Loaded policy for tenant: {tenant_id}")
            except Exception as e:
                logger.error(f"Failed to load policy {policy_file}: {e}")

    def get_policy(self, tenant_id: str) -> dict:
        """Get policy for a tenant, falling back to default."""
        if tenant_id in self._policies:
            return self._policies[tenant_id]
        return self._policies.get("default", {})

    def evaluate(
        self,
        context: EnrichedContext,
        intent_code: str,
        tenant_id: str = "default",
        frustration_score: float = 0.0,
    ) -> PolicyDecision:
        """
        Evaluate policies for the given context and intent.

        Args:
            context: Enriched context with customer and order data.
            intent_code: The resolved intent code (e.g., "RETURN_EXCHANGE.RETURN_INITIATE").
            tenant_id: Tenant identifier for policy lookup.
            frustration_score: Customer frustration score (0-1).

        Returns:
            PolicyDecision with approval/escalation/routing decisions.
        """
        policy = self.get_policy(tenant_id)
        decision = PolicyDecision()

        # Extract components
        customer = context.customer
        order = context.order
        category, intent = self._parse_intent_code(intent_code)

        # Evaluate return eligibility
        if order and category == "RETURN_EXCHANGE":
            self._evaluate_return_eligibility(decision, order, customer, policy)
            decision.rules_applied.append("return_eligibility")

        # Evaluate auto-approval
        if order and customer:
            self._evaluate_auto_approval(decision, order, customer, intent, policy)
            decision.rules_applied.append("auto_approval")

        # Evaluate escalation triggers
        self._evaluate_escalation(
            decision, context, customer, frustration_score, policy
        )
        decision.rules_applied.append("escalation")

        # Evaluate priority routing
        self._evaluate_priority(decision, customer, order, frustration_score, policy)
        decision.rules_applied.append("priority_routing")

        # Generate recommendations
        self._generate_recommendations(decision, intent, order, customer)

        return decision

    def _parse_intent_code(self, intent_code: str) -> tuple[str, str]:
        """Parse intent code into category and intent."""
        parts = intent_code.split(".")
        category = parts[0] if parts else ""
        intent = parts[1] if len(parts) > 1 else ""
        return category, intent

    def _evaluate_return_eligibility(
        self,
        decision: PolicyDecision,
        order: OrderContext,
        customer: CustomerProfile | None,
        policy: dict,
    ) -> None:
        """Evaluate return window and eligibility."""
        return_policy = policy.get("return_policy", {})

        # Check explicit eligibility status
        if order.return_eligibility == ReturnEligibility.FINAL_SALE:
            decision.return_eligible = False
            decision.return_ineligible_reason = "Item is final sale and cannot be returned"
            return

        if order.return_eligibility == ReturnEligibility.EXPIRED:
            decision.return_eligible = False
            decision.return_ineligible_reason = "Return window has expired"
            return

        # Check if within return window
        if not order.is_within_return_window:
            decision.return_eligible = False
            decision.return_ineligible_reason = "Return window has expired"
            return

        # Calculate days remaining
        decision.days_until_return_expires = order.days_until_return_expires

        # Check if order is cancelled
        if order.is_cancelled:
            decision.return_eligible = False
            decision.return_ineligible_reason = "Order has been cancelled"
            return

        # Check final sale categories
        final_sale_categories = return_policy.get("final_sale_categories", [])
        for item in order.items:
            if item.category and item.category.lower() in [c.lower() for c in final_sale_categories]:
                decision.return_eligible = False
                decision.return_ineligible_reason = f"Category '{item.category}' is final sale"
                return

        decision.return_eligible = True

    def _evaluate_auto_approval(
        self,
        decision: PolicyDecision,
        order: OrderContext,
        customer: CustomerProfile,
        intent: str,
        policy: dict,
    ) -> None:
        """Evaluate auto-approval thresholds."""
        auto_approval = policy.get("auto_approval", {})

        if not auto_approval.get("enabled", True):
            return

        tier = customer.tier.value if customer.tier else "standard"
        order_total = order.total

        # Return auto-approval
        if intent in ["RETURN_INITIATE", "RETURN_REQUEST"]:
            return_rules = auto_approval.get("return", {})
            max_amount = return_rules.get(f"max_amount_{tier}", return_rules.get("max_amount_standard", 100))

            if order_total <= max_amount and decision.return_eligible:
                # Check exclusions
                excluded_categories = return_rules.get("excluded_categories", [])
                has_excluded = any(
                    item.category and item.category.lower() in [c.lower() for c in excluded_categories]
                    for item in order.items
                )

                if not has_excluded:
                    decision.auto_approve_return = True

        # Refund auto-approval
        if intent in ["REFUND_STATUS", "REFUND_REQUEST"]:
            refund_rules = auto_approval.get("refund", {})
            max_amount = refund_rules.get(f"max_amount_{tier}", refund_rules.get("max_amount_standard", 50))

            if order_total <= max_amount:
                decision.auto_approve_refund = True

        # Replacement auto-approval
        if intent in ["EXCHANGE_REQUEST", "REPLACEMENT_REQUEST"]:
            replacement_rules = auto_approval.get("replacement", {})
            if replacement_rules.get("enabled", True):
                max_amount = replacement_rules.get("max_amount", 200)
                if order_total <= max_amount:
                    decision.auto_approve_replacement = True

    def _evaluate_escalation(
        self,
        decision: PolicyDecision,
        context: EnrichedContext,
        customer: CustomerProfile | None,
        frustration_score: float,
        policy: dict,
    ) -> None:
        """Evaluate escalation triggers."""
        escalation_rules = policy.get("escalation", {})

        # Check complaint threshold
        if customer:
            complaint_threshold = escalation_rules.get("complaint_threshold", 3)
            if customer.complaints_90d >= complaint_threshold:
                decision.escalation_required = True
                decision.escalation_reasons.append(
                    f"Customer has {customer.complaints_90d} complaints in 90 days (threshold: {complaint_threshold})"
                )

        # Check high-value order
        if context.order:
            high_value_threshold = escalation_rules.get("high_value_threshold", 500)
            if context.order.total >= high_value_threshold:
                decision.escalation_required = True
                decision.escalation_reasons.append(
                    f"High-value order: ${context.order.total:.2f} (threshold: ${high_value_threshold:.2f})"
                )

        # Check frustration score
        frustration_threshold = escalation_rules.get("frustration_score_threshold", 0.7)
        if frustration_score >= frustration_threshold:
            decision.escalation_required = True
            decision.escalation_reasons.append(
                f"High frustration score: {frustration_score:.2f} (threshold: {frustration_threshold})"
            )

        # Check for escalation keywords (would be checked against raw text)
        # This is handled at a higher level where we have access to text

    def check_escalation_keywords(self, text: str, policy: dict) -> list[str]:
        """Check text for escalation-triggering keywords."""
        escalation_rules = policy.get("escalation", {})
        keywords = escalation_rules.get("auto_escalate_keywords", [])
        text_lower = text.lower()

        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)

        return found_keywords

    def _evaluate_priority(
        self,
        decision: PolicyDecision,
        customer: CustomerProfile | None,
        order: OrderContext | None,
        frustration_score: float,
        policy: dict,
    ) -> None:
        """Evaluate priority routing."""
        priority_rules = policy.get("priority_routing", {})

        if not priority_rules.get("enabled", True):
            return

        # VIP priority
        if customer and priority_rules.get("vip_priority", True):
            if customer.tier == CustomerTier.VIP or customer.is_vip:
                decision.priority_flag = True
                decision.priority_reasons.append("VIP customer")

        # High frustration priority
        if priority_rules.get("high_frustration_priority", True):
            threshold = priority_rules.get("frustration_threshold", 0.7)
            if frustration_score >= threshold:
                decision.priority_flag = True
                decision.priority_reasons.append(f"High frustration ({frustration_score:.2f})")

        # High-value order priority
        if order:
            threshold = priority_rules.get("high_value_order_threshold", 300)
            if order.total >= threshold:
                decision.priority_flag = True
                decision.priority_reasons.append(f"High-value order (${order.total:.2f})")

    def _generate_recommendations(
        self,
        decision: PolicyDecision,
        intent: str,
        order: OrderContext | None,
        customer: CustomerProfile | None,
    ) -> None:
        """Generate action recommendations."""
        if decision.auto_approve_return:
            decision.recommended_action = "auto_approve_return"
            decision.suggested_resolution = "Issue return label and initiate return process"
        elif decision.auto_approve_refund:
            decision.recommended_action = "auto_approve_refund"
            decision.suggested_resolution = "Process refund to original payment method"
        elif decision.auto_approve_replacement:
            decision.recommended_action = "auto_approve_replacement"
            decision.suggested_resolution = "Send replacement item and issue return label"
        elif decision.escalation_required:
            decision.recommended_action = "escalate_to_supervisor"
            decision.suggested_resolution = "Route to supervisor for review"
        elif not decision.return_eligible and intent in ["RETURN_INITIATE", "RETURN_REQUEST"]:
            decision.recommended_action = "explain_policy"
            decision.suggested_resolution = decision.return_ineligible_reason
        else:
            decision.recommended_action = "agent_review"
            decision.suggested_resolution = "Review case and apply appropriate resolution"

    def validate_return_window(
        self,
        order: OrderContext,
        customer_tier: CustomerTier = CustomerTier.STANDARD,
        tenant_id: str = "default",
    ) -> tuple[bool, str | None, int | None]:
        """
        Validate if an order is within return window.

        Args:
            order: Order context.
            customer_tier: Customer tier for extended windows.
            tenant_id: Tenant for policy lookup.

        Returns:
            Tuple of (is_eligible, reason_if_not, days_remaining).
        """
        policy = self.get_policy(tenant_id)
        return_policy = policy.get("return_policy", {})

        # Get window based on tier
        if customer_tier == CustomerTier.VIP:
            window_days = return_policy.get("vip_window_days", 60)
        elif customer_tier == CustomerTier.PREMIUM:
            window_days = return_policy.get("premium_window_days", 45)
        else:
            window_days = return_policy.get("default_window_days", 30)

        # Calculate eligibility
        if not order.created_at:
            return True, None, None

        window_end = order.created_at + timedelta(days=window_days)
        now = datetime.utcnow()
        remaining = (window_end - now).days

        if remaining < 0:
            return False, f"Return window expired {abs(remaining)} days ago", remaining

        return True, None, remaining


# Singleton instance
_default_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Get or create the default policy engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = PolicyEngine()
    return _default_engine
