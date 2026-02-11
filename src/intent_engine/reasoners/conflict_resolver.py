"""Conflict resolution for contradictory intents in compound requests."""

import re

from intent_engine.models.conflict import (
    ConflictResolutionOutput,
    ConflictType,
    ResolutionStrategy,
)
from intent_engine.models.context import EnrichedContext
from intent_engine.models.entity import ExtractedEntity
from intent_engine.models.intent import ResolvedIntent
from intent_engine.models.response import Constraint


class ConflictResolver:
    """
    Resolve conflicts between contradictory intents in compound requests.

    Sits between IntentDecomposer and PolicyEngine in the processing pipeline.
    Detects mutually exclusive intents (e.g., return AND exchange for same item)
    and resolves them based on customer preference, business rules, or escalation.
    """

    # Conflict detection matrix: (intent_a, intent_b) -> conflict_description
    # Order doesn't matter - both directions are checked
    EXCLUSIVE_PAIRS: dict[tuple[str, str], str] = {
        # Return vs Exchange for same item
        ("RETURN_EXCHANGE.RETURN_INITIATE", "RETURN_EXCHANGE.EXCHANGE_REQUEST"): (
            "Cannot both return and exchange the same item"
        ),
        ("RETURN_EXCHANGE.REFUND_STATUS", "RETURN_EXCHANGE.EXCHANGE_REQUEST"): (
            "Cannot request refund and exchange for the same item"
        ),
        # Cancel vs Modify
        ("ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.EXPEDITE"): (
            "Cannot cancel and expedite the same order"
        ),
        ("ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.CHANGE_ADDRESS"): (
            "Cannot cancel and change address for the same order"
        ),
        ("ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.CHANGE_ITEMS"): (
            "Cannot cancel and modify items for the same order"
        ),
        # Other contradictory actions
        ("ORDER_MODIFY.EXPEDITE", "ORDER_MODIFY.DELAY_SHIPMENT"): (
            "Cannot expedite and delay the same shipment"
        ),
    }

    # Non-conflicting pairs (informational - these are complementary)
    COMPLEMENTARY_PAIRS: set[tuple[str, str]] = {
        ("ORDER_STATUS.WISMO", "ORDER_STATUS.DELIVERY_ESTIMATE"),
        ("RETURN_EXCHANGE.RETURN_INITIATE", "RETURN_EXCHANGE.REFUND_STATUS"),
        ("COMPLAINT.DAMAGED_ITEM", "RETURN_EXCHANGE.RETURN_INITIATE"),
        ("COMPLAINT.WRONG_ITEM", "RETURN_EXCHANGE.EXCHANGE_REQUEST"),
    }

    # Priority rankings for conflict resolution (higher = preferred)
    # Merchant typically prefers exchange over refund (keeps customer)
    INTENT_PRIORITY: dict[str, int] = {
        "RETURN_EXCHANGE.EXCHANGE_REQUEST": 3,
        "RETURN_EXCHANGE.RETURN_INITIATE": 2,
        "RETURN_EXCHANGE.REFUND_STATUS": 1,
        "ORDER_MODIFY.EXPEDITE": 2,
        "ORDER_MODIFY.CANCEL_ORDER": 1,
    }

    # Keywords indicating customer preference
    PREFERENCE_KEYWORDS: dict[str, list[str]] = {
        "exchange": ["exchange", "swap", "different size", "different color", "replace with"],
        "refund": ["refund", "money back", "return for refund", "just return"],
        "cancel": ["cancel", "don't want", "changed my mind"],
        "expedite": ["faster", "rush", "expedite", "urgent", "asap"],
    }

    async def resolve(
        self,
        intents: list[ResolvedIntent],
        entities: list[ExtractedEntity],
        context: EnrichedContext | None = None,
        constraints: list[Constraint] | None = None,
        text: str = "",
        customer_tier: str | None = None,
        frustration_score: float = 0.0,
    ) -> ConflictResolutionOutput:
        """
        Resolve conflicts between intents.

        Args:
            intents: List of resolved intents from decomposition.
            entities: Extracted entities from the message.
            context: Enriched context with customer/order data.
            constraints: Constraints from decomposition.
            text: Original customer message text.
            customer_tier: Customer tier (VIP, standard, etc.).
            frustration_score: Frustration level from sentiment analysis.

        Returns:
            ConflictResolutionOutput with resolved intents and reasoning.
        """
        reasoning: list[str] = ["Step 9: Conflict resolution"]

        if len(intents) < 2:
            reasoning.append("Single intent - no conflict possible")
            return ConflictResolutionOutput(
                resolved_intents=intents,
                has_conflict=False,
                reasoning=reasoning,
            )

        # Detect conflicts
        conflicts = self._detect_conflicts(intents)

        if not conflicts:
            reasoning.append(f"No conflicts detected among {len(intents)} intents")
            return ConflictResolutionOutput(
                resolved_intents=intents,
                has_conflict=False,
                reasoning=reasoning,
            )

        # We have conflicts - handle them
        intent_a, intent_b, conflict_desc = conflicts[0]  # Handle first conflict
        reasoning.append(f"Detected conflict: {intent_a.intent_code} vs {intent_b.intent_code}")
        reasoning.append(f"Conflict type: {conflict_desc}")

        # Determine conflict type
        conflict_type = self._determine_conflict_type(intent_a, intent_b, context)

        # Check if items are different (no conflict if different items)
        if self._check_different_items(intents, entities):
            reasoning.append("Intents apply to different items - no conflict")
            return ConflictResolutionOutput(
                resolved_intents=intents,
                has_conflict=False,
                reasoning=reasoning,
            )

        # Check for explicit customer preference
        preference = self._extract_preference(text)
        if preference:
            reasoning.append(f"Customer preference detected: '{preference}'")
            resolved = self._apply_preference(intents, preference, intent_a, intent_b)
            if resolved:
                reasoning.append(
                    f"Resolved to {resolved[0].intent_code} based on stated preference"
                )
                return ConflictResolutionOutput(
                    resolved_intents=resolved,
                    has_conflict=True,
                    conflict_type=conflict_type,
                    conflict_description=conflict_desc,
                    resolution_strategy=ResolutionStrategy.PREFERENCE,
                    reasoning=reasoning,
                )

        # Check for VIP/AT_RISK leniency - may allow both
        if customer_tier and customer_tier.lower() in ("vip", "at_risk"):
            can_approve_both = self._can_approve_both_for_tier(intent_a, intent_b, customer_tier)
            if can_approve_both:
                reasoning.append(f"{customer_tier.upper()} customer - approving both actions")
                return ConflictResolutionOutput(
                    resolved_intents=intents,
                    has_conflict=True,
                    conflict_type=conflict_type,
                    conflict_description=conflict_desc,
                    resolution_strategy=ResolutionStrategy.PRIORITY,
                    reasoning=reasoning,
                )

        # Check frustration score - favor customer-friendly option
        if frustration_score > 0.7:
            reasoning.append(
                f"High frustration ({frustration_score:.2f}) - favoring customer preference"
            )
            resolved = self._apply_customer_favorable(intents, intent_a, intent_b)
            reasoning.append(f"Resolved to {resolved[0].intent_code} (customer-favorable)")
            return ConflictResolutionOutput(
                resolved_intents=resolved,
                has_conflict=True,
                conflict_type=conflict_type,
                conflict_description=conflict_desc,
                resolution_strategy=ResolutionStrategy.PRIORITY,
                reasoning=reasoning,
            )

        # Apply business priority rules
        resolved = self._apply_priority_rules(intents, intent_a, intent_b)
        if resolved:
            reasoning.append(f"Applied business priority: {resolved[0].intent_code} preferred")
            return ConflictResolutionOutput(
                resolved_intents=resolved,
                has_conflict=True,
                conflict_type=conflict_type,
                conflict_description=conflict_desc,
                resolution_strategy=ResolutionStrategy.PRIORITY,
                reasoning=reasoning,
            )

        # No clear resolution - need clarification
        question, options = self._generate_clarification(intent_a, intent_b)
        reasoning.append("No clear resolution - requesting clarification")

        return ConflictResolutionOutput(
            resolved_intents=intents,  # Keep both until clarified
            has_conflict=True,
            conflict_type=conflict_type,
            conflict_description=conflict_desc,
            resolution_strategy=ResolutionStrategy.CLARIFICATION,
            reasoning=reasoning,
            requires_clarification=True,
            clarification_question=question,
            clarification_options=options,
        )

    def _detect_conflicts(
        self, intents: list[ResolvedIntent]
    ) -> list[tuple[ResolvedIntent, ResolvedIntent, str]]:
        """Detect conflicts between pairs of intents."""
        conflicts = []

        for i, intent_a in enumerate(intents):
            for intent_b in intents[i + 1 :]:
                code_a = intent_a.intent_code
                code_b = intent_b.intent_code

                # Check both orderings
                conflict_desc = self.EXCLUSIVE_PAIRS.get(
                    (code_a, code_b)
                ) or self.EXCLUSIVE_PAIRS.get((code_b, code_a))

                if conflict_desc:
                    conflicts.append((intent_a, intent_b, conflict_desc))

        return conflicts

    def _determine_conflict_type(
        self,
        intent_a: ResolvedIntent,
        intent_b: ResolvedIntent,
        context: EnrichedContext | None,
    ) -> ConflictType:
        """Determine the type of conflict."""
        # Check for policy violation (e.g., return after window expired)
        if context and context.order:
            if not context.order.is_within_return_window:
                if any("RETURN" in i.intent_code for i in [intent_a, intent_b]):
                    return ConflictType.POLICY_VIOLATION

        # Check for contradictory policy (actions that logically cannot coexist)
        codes = {intent_a.intent_code, intent_b.intent_code}
        contradictory_pairs = {
            frozenset(["ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.EXPEDITE"]),
            frozenset(["ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.CHANGE_ADDRESS"]),
            frozenset(["ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.CHANGE_ITEMS"]),
            frozenset(["ORDER_MODIFY.EXPEDITE", "ORDER_MODIFY.DELAY_SHIPMENT"]),
        }
        if frozenset(codes) in contradictory_pairs:
            return ConflictType.CONTRADICTORY_POLICY

        # Default: mutually exclusive
        return ConflictType.MUTUALLY_EXCLUSIVE

    def _check_different_items(
        self, intents: list[ResolvedIntent], entities: list[ExtractedEntity]
    ) -> bool:
        """Check if intents apply to different items (no conflict)."""
        # Look for multiple product SKUs or order IDs in entities
        product_entities = [
            e for e in entities if e.entity_type.value in ("product_sku", "product_id")
        ]

        # If multiple different products mentioned, might not be a conflict
        if len(product_entities) >= 2:
            unique_products = set(e.value for e in product_entities)
            if len(unique_products) >= 2:
                return True

        return False

    def _extract_preference(self, text: str) -> str | None:
        """Extract customer preference from text."""
        text_lower = text.lower()

        # Check for explicit preference patterns with action words
        preference_patterns = [
            r"(?:i\s+)?prefer\s+(?:to\s+)?(\w+)",
            r"(?:i\s+)?(?:want|would like)\s+(?:to\s+)?(?:a\s+)?(\w+)\s+(?:not|instead)",
            r"(?:just|only)\s+(?:want\s+(?:to\s+)?)?(?:a\s+)?(\w+)",
            r"(\w+)\s+not\s+(?:a\s+)?(?:refund|return|exchange)",
        ]

        for pattern in preference_patterns:
            match = re.search(pattern, text_lower)
            if match:
                word = match.group(1)
                # Map to preference category
                for pref_type, keywords in self.PREFERENCE_KEYWORDS.items():
                    if any(kw in word or word in kw for kw in keywords):
                        return pref_type

        # Check for negation pattern: "X, not Y" - prefer what's NOT negated
        # "refund, not exchange" -> prefer refund
        # "return for a refund, not exchange" -> prefer refund
        negation_match = re.search(
            r"(refund|return|exchange|cancel|expedite)[^,]*,?\s*not\s+(refund|return|exchange|cancel|expedite)",
            text_lower,
        )
        if negation_match:
            preferred_word = negation_match.group(1)
            for pref_type, keywords in self.PREFERENCE_KEYWORDS.items():
                if any(kw in preferred_word or preferred_word in kw for kw in keywords):
                    return pref_type

        return None

    def _apply_preference(
        self,
        intents: list[ResolvedIntent],
        preference: str,
        conflict_a: ResolvedIntent,
        conflict_b: ResolvedIntent,
    ) -> list[ResolvedIntent] | None:
        """Apply customer preference to resolve conflict."""
        preference_intent_map = {
            "exchange": "EXCHANGE_REQUEST",
            "refund": "RETURN_INITIATE",
            "cancel": "CANCEL_ORDER",
            "expedite": "EXPEDITE",
        }

        preferred_intent_name = preference_intent_map.get(preference)
        if not preferred_intent_name:
            return None

        # Find the preferred intent among conflicting ones
        preferred = None
        for intent in [conflict_a, conflict_b]:
            if preferred_intent_name in intent.intent_code:
                preferred = intent
                break

        if not preferred:
            return None

        # Keep preferred intent, remove the other conflicting one
        removed_code = conflict_b.intent_code if preferred == conflict_a else conflict_a.intent_code
        return [i for i in intents if i.intent_code != removed_code]

    def _can_approve_both_for_tier(
        self,
        intent_a: ResolvedIntent,
        intent_b: ResolvedIntent,
        tier: str,
    ) -> bool:
        """Check if both actions can be approved for VIP/at-risk customers."""
        # For VIP/at-risk, we might allow both return AND exchange
        # (e.g., return one item, exchange another)
        # This is a business decision - being lenient with high-value customers

        # Don't allow truly contradictory actions even for VIP
        codes = {intent_a.intent_code, intent_b.intent_code}
        contradictory_pairs = {
            frozenset(["ORDER_MODIFY.CANCEL_ORDER", "ORDER_MODIFY.EXPEDITE"]),
            frozenset(["ORDER_MODIFY.EXPEDITE", "ORDER_MODIFY.DELAY_SHIPMENT"]),
        }

        if frozenset(codes) in contradictory_pairs:
            return False

        # VIP/AT_RISK can do return + exchange (we'll handle as sequential)
        return tier.lower() in ("vip", "at_risk")

    def _apply_customer_favorable(
        self,
        intents: list[ResolvedIntent],
        conflict_a: ResolvedIntent,
        conflict_b: ResolvedIntent,
    ) -> list[ResolvedIntent]:
        """Apply customer-favorable resolution for high frustration."""
        # For high frustration, prefer the option that gives customer what they want most
        # Generally: refund > exchange > other
        customer_favorable_priority = {
            "RETURN_INITIATE": 3,
            "REFUND_STATUS": 3,
            "EXCHANGE_REQUEST": 2,
            "CANCEL_ORDER": 2,
        }

        score_a = customer_favorable_priority.get(conflict_a.intent, 0)
        score_b = customer_favorable_priority.get(conflict_b.intent, 0)

        removed = conflict_b if score_a >= score_b else conflict_a
        return [i for i in intents if i.intent_code != removed.intent_code]

    def _apply_priority_rules(
        self,
        intents: list[ResolvedIntent],
        conflict_a: ResolvedIntent,
        conflict_b: ResolvedIntent,
    ) -> list[ResolvedIntent] | None:
        """Apply business priority rules to resolve conflict."""
        priority_a = self.INTENT_PRIORITY.get(conflict_a.intent_code, 0)
        priority_b = self.INTENT_PRIORITY.get(conflict_b.intent_code, 0)

        if priority_a == priority_b:
            return None  # Can't resolve by priority alone

        # Keep higher priority, remove lower
        removed = conflict_b if priority_a > priority_b else conflict_a
        return [i for i in intents if i.intent_code != removed.intent_code]

    def _generate_clarification(
        self,
        intent_a: ResolvedIntent,
        intent_b: ResolvedIntent,
    ) -> tuple[str, list[str]]:
        """Generate a clarification question for unresolved conflicts."""
        intent_descriptions = {
            "RETURN_INITIATE": "return the item for a refund",
            "EXCHANGE_REQUEST": "exchange the item for a different one",
            "REFUND_STATUS": "get a refund",
            "CANCEL_ORDER": "cancel your order",
            "EXPEDITE": "expedite shipping",
            "CHANGE_ADDRESS": "change the shipping address",
        }

        desc_a = intent_descriptions.get(intent_a.intent, intent_a.intent.lower().replace("_", " "))
        desc_b = intent_descriptions.get(intent_b.intent, intent_b.intent.lower().replace("_", " "))

        question = (
            f"I noticed you'd like to both {desc_a} and {desc_b}. "
            f"These options are mutually exclusive for the same item. "
            f"Which would you prefer?"
        )

        options = [
            f"I'd like to {desc_a}",
            f"I'd like to {desc_b}",
            "I need help deciding",
        ]

        return question, options
