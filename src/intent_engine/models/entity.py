"""Entity extraction models."""

from enum import Enum

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities that can be extracted from customer messages."""

    ORDER_ID = "order_id"
    PRODUCT_SKU = "product_sku"
    PRODUCT_NAME = "product_name"
    TRACKING_NUMBER = "tracking_number"
    DATE = "date"
    DEADLINE = "deadline"  # "by Friday", "within 2 days"
    MONEY_AMOUNT = "money_amount"
    SIZE = "size"
    COLOR = "color"
    QUANTITY = "quantity"
    ADDRESS = "address"
    PERSON_NAME = "person_name"
    REASON = "reason"  # Return reason, complaint reason
    EMAIL = "email"
    PHONE = "phone"

    # Phase 5: Advanced entity types
    DAMAGE_SEVERITY = "damage_severity"  # "slight scratch", "completely broken"
    DEFECT_CATEGORY = "defect_category"  # "color_mismatch", "size_wrong", "broken"
    BRAND_NAME = "brand_name"  # Product brand for lookups
    CARRIER = "carrier"  # "UPS", "FedEx", "USPS"
    COMPLAINT_REASON = "complaint_reason"  # Structured reason category


class DamageSeverity(str, Enum):
    """Severity levels for damage reports."""

    MINOR = "minor"  # Cosmetic, doesn't affect function
    MODERATE = "moderate"  # Noticeable, partial function
    SEVERE = "severe"  # Major damage, non-functional
    DESTROYED = "destroyed"  # Complete loss


class DefectCategory(str, Enum):
    """Categories of product defects."""

    COLOR_MISMATCH = "color_mismatch"
    SIZE_WRONG = "size_wrong"
    BROKEN = "broken"
    MISSING_PARTS = "missing_parts"
    MANUFACTURING_DEFECT = "manufacturing_defect"
    SHIPPING_DAMAGE = "shipping_damage"
    NOT_AS_DESCRIBED = "not_as_described"
    FUNCTIONALITY_ISSUE = "functionality_issue"


class ExtractedEntity(BaseModel):
    """A single extracted entity from the input text."""

    entity_type: EntityType
    value: str
    raw_span: str = Field(description="Original text span")
    start_pos: int
    end_pos: int
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = {"json_schema_extra": {"examples": [
        {
            "entity_type": "order_id",
            "value": "ORD-98765",
            "raw_span": "#ORD-98765",
            "start_pos": 18,
            "end_pos": 28,
            "confidence": 0.99,
        }
    ]}}


class ExtractionResult(BaseModel):
    """Complete extraction results from input processing."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    frustration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority_flag: bool = Field(
        default=False,
        description="True if message should be routed to priority queue",
    )
    sentiment_signals: list[str] = Field(
        default_factory=list,
        description="Detected sentiment/frustration signals",
    )
    embedding: list[float] = Field(
        default_factory=list,
        description="Semantic embedding vector (384 dims for MiniLM)",
    )
