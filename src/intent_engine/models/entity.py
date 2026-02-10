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
    embedding: list[float] = Field(
        default_factory=list,
        description="Semantic embedding vector (384 dims for MiniLM)",
    )
