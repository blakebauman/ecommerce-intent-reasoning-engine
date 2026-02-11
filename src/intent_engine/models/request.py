"""Input request models for the intent engine."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InputChannel(str, Enum):
    """Supported input channels."""

    CHAT = "chat"
    EMAIL = "email"
    FORM = "form"
    WEBHOOK = "webhook"
    SMS = "sms"
    SOCIAL = "social"
    CLICKSTREAM = "clickstream"


class Attachment(BaseModel):
    """File attachment from customer input."""

    url: str
    mime_type: str
    filename: str | None = None
    analysis: dict[str, Any] | None = None  # Populated by attachment processor


class IntentRequest(BaseModel):
    """Unified input model - every channel adapter produces this."""

    request_id: str = Field(description="Unique request identifier")
    tenant_id: str = Field(description="Merchant/store identifier")
    channel: InputChannel
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Raw input
    raw_text: str = Field(description="Primary text content")
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Channel-specific metadata (email headers, form fields, etc.)",
    )
    attachments: list[Attachment] = Field(default_factory=list)

    # Conversation context
    conversation_id: str | None = None
    message_index: int = 0  # Position in multi-turn conversation
    previous_intents: list[str] = Field(
        default_factory=list,
        description="Intents resolved earlier in this conversation",
    )

    # Customer context (populated by enrichment)
    customer_id: str | None = None
    customer_tier: str | None = None  # VIP, standard, new, flagged
    order_ids: list[str] = Field(
        default_factory=list,
        description="Order IDs mentioned or inferred",
    )

    # Behavioral signals
    page_context: str | None = None  # URL/page the customer was on
    session_actions: list[str] = Field(
        default_factory=list,
        description="Recent clickstream actions",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req-12345",
                    "tenant_id": "merchant-1",
                    "channel": "chat",
                    "raw_text": "Where is my order #ORD-98765?",
                    "timestamp": "2024-02-09T10:30:00Z",
                }
            ]
        }
    }
