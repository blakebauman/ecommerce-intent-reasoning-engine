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
    The 8 core intents for MVP.

    Format: CATEGORY.INTENT
    """

    # Order Status
    ORDER_STATUS_WISMO = "ORDER_STATUS.WISMO"
    ORDER_STATUS_DELIVERY_ESTIMATE = "ORDER_STATUS.DELIVERY_ESTIMATE"

    # Order Modification
    ORDER_MODIFY_CANCEL_ORDER = "ORDER_MODIFY.CANCEL_ORDER"
    ORDER_MODIFY_CHANGE_ADDRESS = "ORDER_MODIFY.CHANGE_ADDRESS"

    # Returns & Exchanges
    RETURN_EXCHANGE_RETURN_INITIATE = "RETURN_EXCHANGE.RETURN_INITIATE"
    RETURN_EXCHANGE_EXCHANGE_REQUEST = "RETURN_EXCHANGE.EXCHANGE_REQUEST"
    RETURN_EXCHANGE_REFUND_STATUS = "RETURN_EXCHANGE.REFUND_STATUS"

    # Complaints
    COMPLAINT_DAMAGED_ITEM = "COMPLAINT.DAMAGED_ITEM"

    @property
    def category(self) -> str:
        """Get the category portion of the intent code."""
        return self.value.split(".")[0]

    @property
    def intent_name(self) -> str:
        """Get the intent name portion of the intent code."""
        return self.value.split(".")[1]


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
