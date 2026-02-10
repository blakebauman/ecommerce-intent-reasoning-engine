"""Intent classification models."""

from enum import Enum

from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    """Primary intent categories."""

    ORDER_STATUS = "ORDER_STATUS"
    ORDER_MODIFY = "ORDER_MODIFY"
    RETURN_EXCHANGE = "RETURN_EXCHANGE"
    COMPLAINT = "COMPLAINT"
    PRODUCT_INQUIRY = "PRODUCT_INQUIRY"
    ACCOUNT_BILLING = "ACCOUNT_BILLING"
    DISCOVERY = "DISCOVERY"
    META = "META"


class CoreIntent(str, Enum):
    """
    Full intent taxonomy for Phase 2.

    Format: CATEGORY.INTENT

    Categories:
    - ORDER_STATUS: Tracking, delivery estimates, confirmation
    - ORDER_MODIFY: Cancellation, address changes, item changes
    - RETURN_EXCHANGE: Returns, exchanges, refunds, warranties
    - COMPLAINT: Damaged items, wrong items, service issues
    - PRODUCT_INQUIRY: Stock, sizing, features, compatibility
    - ACCOUNT_BILLING: Payment, subscription, account management
    - DISCOVERY: Product recommendations, comparisons
    - META: Greetings, handoffs, feedback
    """

    # =========================================================================
    # ORDER_STATUS - Order tracking and status inquiries
    # =========================================================================
    ORDER_STATUS_WISMO = "ORDER_STATUS.WISMO"  # Where Is My Order
    ORDER_STATUS_DELIVERY_ESTIMATE = "ORDER_STATUS.DELIVERY_ESTIMATE"
    ORDER_STATUS_CONFIRMATION = "ORDER_STATUS.CONFIRMATION"  # Did my order go through?
    ORDER_STATUS_TRACKING_ISSUE = "ORDER_STATUS.TRACKING_ISSUE"  # Tracking not updating

    # =========================================================================
    # ORDER_MODIFY - Order modification requests
    # =========================================================================
    ORDER_MODIFY_CANCEL_ORDER = "ORDER_MODIFY.CANCEL_ORDER"
    ORDER_MODIFY_CHANGE_ADDRESS = "ORDER_MODIFY.CHANGE_ADDRESS"
    ORDER_MODIFY_CHANGE_ITEMS = "ORDER_MODIFY.CHANGE_ITEMS"  # Add/remove/change items
    ORDER_MODIFY_CHANGE_PAYMENT = "ORDER_MODIFY.CHANGE_PAYMENT"  # Change payment method
    ORDER_MODIFY_EXPEDITE = "ORDER_MODIFY.EXPEDITE"  # Upgrade shipping speed

    # =========================================================================
    # RETURN_EXCHANGE - Returns, exchanges, and refunds
    # =========================================================================
    RETURN_EXCHANGE_RETURN_INITIATE = "RETURN_EXCHANGE.RETURN_INITIATE"
    RETURN_EXCHANGE_EXCHANGE_REQUEST = "RETURN_EXCHANGE.EXCHANGE_REQUEST"
    RETURN_EXCHANGE_REFUND_STATUS = "RETURN_EXCHANGE.REFUND_STATUS"
    RETURN_EXCHANGE_RETURN_STATUS = "RETURN_EXCHANGE.RETURN_STATUS"  # Where is my return?
    RETURN_EXCHANGE_RETURN_POLICY = "RETURN_EXCHANGE.RETURN_POLICY"  # Policy questions
    RETURN_EXCHANGE_WARRANTY_CLAIM = "RETURN_EXCHANGE.WARRANTY_CLAIM"  # Warranty issues
    RETURN_EXCHANGE_STORE_CREDIT = "RETURN_EXCHANGE.STORE_CREDIT"  # Store credit inquiries

    # =========================================================================
    # COMPLAINT - Issues and complaints
    # =========================================================================
    COMPLAINT_DAMAGED_ITEM = "COMPLAINT.DAMAGED_ITEM"
    COMPLAINT_WRONG_ITEM = "COMPLAINT.WRONG_ITEM"  # Received wrong product
    COMPLAINT_MISSING_ITEM = "COMPLAINT.MISSING_ITEM"  # Item missing from order
    COMPLAINT_QUALITY_ISSUE = "COMPLAINT.QUALITY_ISSUE"  # Poor quality, not as described
    COMPLAINT_LATE_DELIVERY = "COMPLAINT.LATE_DELIVERY"  # Delivery took too long
    COMPLAINT_SERVICE_ISSUE = "COMPLAINT.SERVICE_ISSUE"  # Bad customer service experience
    COMPLAINT_LOST_PACKAGE = "COMPLAINT.LOST_PACKAGE"  # Package lost in transit

    # =========================================================================
    # PRODUCT_INQUIRY - Product questions
    # =========================================================================
    PRODUCT_INQUIRY_STOCK = "PRODUCT_INQUIRY.STOCK"  # Is it in stock?
    PRODUCT_INQUIRY_RESTOCK = "PRODUCT_INQUIRY.RESTOCK"  # When will it be back?
    PRODUCT_INQUIRY_SIZE_FIT = "PRODUCT_INQUIRY.SIZE_FIT"  # Sizing questions
    PRODUCT_INQUIRY_FEATURES = "PRODUCT_INQUIRY.FEATURES"  # Product features/specs
    PRODUCT_INQUIRY_COMPATIBILITY = "PRODUCT_INQUIRY.COMPATIBILITY"  # Will it work with X?
    PRODUCT_INQUIRY_MATERIALS = "PRODUCT_INQUIRY.MATERIALS"  # What is it made of?
    PRODUCT_INQUIRY_USAGE = "PRODUCT_INQUIRY.USAGE"  # How do I use it?
    PRODUCT_INQUIRY_AUTHENTICITY = "PRODUCT_INQUIRY.AUTHENTICITY"  # Is it genuine?

    # =========================================================================
    # ACCOUNT_BILLING - Account and billing
    # =========================================================================
    ACCOUNT_BILLING_PAYMENT_ISSUE = "ACCOUNT_BILLING.PAYMENT_ISSUE"  # Payment failed
    ACCOUNT_BILLING_UPDATE_PAYMENT = "ACCOUNT_BILLING.UPDATE_PAYMENT"  # Change payment
    ACCOUNT_BILLING_INVOICE = "ACCOUNT_BILLING.INVOICE"  # Need invoice/receipt
    ACCOUNT_BILLING_PROMO_CODE = "ACCOUNT_BILLING.PROMO_CODE"  # Discount code issues
    ACCOUNT_BILLING_SUBSCRIPTION = "ACCOUNT_BILLING.SUBSCRIPTION"  # Subscription management
    ACCOUNT_BILLING_ACCOUNT_ACCESS = "ACCOUNT_BILLING.ACCOUNT_ACCESS"  # Login issues
    ACCOUNT_BILLING_UPDATE_INFO = "ACCOUNT_BILLING.UPDATE_INFO"  # Update account details
    ACCOUNT_BILLING_DELETE_ACCOUNT = "ACCOUNT_BILLING.DELETE_ACCOUNT"  # Account deletion

    # =========================================================================
    # DISCOVERY - Product discovery and recommendations
    # =========================================================================
    DISCOVERY_RECOMMENDATION = "DISCOVERY.RECOMMENDATION"  # Suggest products
    DISCOVERY_COMPARISON = "DISCOVERY.COMPARISON"  # Compare products
    DISCOVERY_BEST_SELLER = "DISCOVERY.BEST_SELLER"  # What's popular?
    DISCOVERY_NEW_ARRIVALS = "DISCOVERY.NEW_ARRIVALS"  # What's new?
    DISCOVERY_GIFT_SUGGESTION = "DISCOVERY.GIFT_SUGGESTION"  # Gift ideas
    DISCOVERY_BUNDLE = "DISCOVERY.BUNDLE"  # Bundle deals

    # =========================================================================
    # META - Conversation management
    # =========================================================================
    META_GREETING = "META.GREETING"  # Hello, hi
    META_FAREWELL = "META.FAREWELL"  # Goodbye, thanks
    META_HUMAN_HANDOFF = "META.HUMAN_HANDOFF"  # Talk to a person
    META_FEEDBACK = "META.FEEDBACK"  # Feedback about service
    META_UNCLEAR = "META.UNCLEAR"  # Cannot understand request
    META_OFF_TOPIC = "META.OFF_TOPIC"  # Not eCommerce related

    @property
    def category(self) -> str:
        """Get the category portion of the intent code."""
        return self.value.split(".")[0]

    @property
    def intent_name(self) -> str:
        """Get the intent name portion of the intent code."""
        return self.value.split(".")[1]


# Mapping for backward compatibility and quick lookup
INTENT_CODES = {intent.value: intent for intent in CoreIntent}


def get_intent_by_code(code: str) -> CoreIntent | None:
    """Get CoreIntent by its code string."""
    return INTENT_CODES.get(code)


def get_intents_by_category(category: str) -> list[CoreIntent]:
    """Get all intents in a category."""
    return [intent for intent in CoreIntent if intent.category == category]


class IntentConfidence(str, Enum):
    """Confidence tier for intent classification."""

    HIGH = "high"  # > 0.85 - fast path, auto-resolve
    MEDIUM = "medium"  # 0.60 - 0.85 - reasoning path
    LOW = "low"  # < 0.60 - needs clarification or human handoff


class ResolvedIntent(BaseModel):
    """Single atomic intent with confidence."""

    category: str = Field(description="Intent category (e.g., ORDER_STATUS)")
    intent: str = Field(description="Specific intent (e.g., WISMO)")
    sub_intent: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_tier: IntentConfidence
    evidence: list[str] = Field(
        default_factory=list,
        description="Text spans or signals that support this classification",
    )

    @classmethod
    def from_core_intent(
        cls,
        core_intent: CoreIntent,
        confidence: float,
        evidence: list[str] | None = None,
    ) -> "ResolvedIntent":
        """Create a ResolvedIntent from a CoreIntent enum."""
        if confidence >= 0.85:
            tier = IntentConfidence.HIGH
        elif confidence >= 0.60:
            tier = IntentConfidence.MEDIUM
        else:
            tier = IntentConfidence.LOW

        return cls(
            category=core_intent.category,
            intent=core_intent.intent_name,
            confidence=confidence,
            confidence_tier=tier,
            evidence=evidence or [],
        )

    @property
    def intent_code(self) -> str:
        """Get the full intent code (CATEGORY.INTENT)."""
        return f"{self.category}.{self.intent}"

    model_config = {"json_schema_extra": {"examples": [
        {
            "category": "ORDER_STATUS",
            "intent": "WISMO",
            "confidence": 0.92,
            "confidence_tier": "high",
            "evidence": ["where is my order"],
        }
    ]}}
