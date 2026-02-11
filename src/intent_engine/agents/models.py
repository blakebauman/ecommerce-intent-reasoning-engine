"""Models for the orchestration agent."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(StrEnum):
    """Types of actions the agent can recommend or take."""

    # Informational
    PROVIDE_ORDER_STATUS = "provide_order_status"
    PROVIDE_TRACKING_INFO = "provide_tracking_info"
    PROVIDE_DELIVERY_ESTIMATE = "provide_delivery_estimate"
    PROVIDE_REFUND_STATUS = "provide_refund_status"

    # Order modifications
    INITIATE_CANCELLATION = "initiate_cancellation"
    UPDATE_SHIPPING_ADDRESS = "update_shipping_address"

    # Returns & exchanges
    INITIATE_RETURN = "initiate_return"
    INITIATE_EXCHANGE = "initiate_exchange"
    GENERATE_RETURN_LABEL = "generate_return_label"

    # Complaints
    ESCALATE_TO_HUMAN = "escalate_to_human"
    CREATE_SUPPORT_TICKET = "create_support_ticket"
    OFFER_COMPENSATION = "offer_compensation"

    # Clarification
    REQUEST_ORDER_ID = "request_order_id"
    REQUEST_CLARIFICATION = "request_clarification"

    # Loyalty and shipping (post-purchase extensions)
    PROVIDE_POINTS_BALANCE = "provide_points_balance"
    PROVIDE_MEMBER_STATUS = "provide_member_status"
    PROVIDE_SHIPPING_OPTIONS = "provide_shipping_options"

    # No action needed
    NONE = "none"


class AgentAction(BaseModel):
    """An action the agent recommends or will take."""

    action_type: ActionType
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    auto_execute: bool = False


class CustomerMessage(BaseModel):
    """Input message from a customer."""

    message_id: str = Field(description="Unique message identifier")
    conversation_id: str | None = Field(default=None, description="Conversation thread ID")
    customer_email: str | None = Field(
        default=None, description="Customer email for context lookup"
    )
    customer_id: str | None = Field(default=None, description="Customer ID if known")
    text: str = Field(description="The customer's message text")
    channel: str = Field(default="chat", description="Input channel (chat, email, etc.)")
    platform: str | None = Field(
        default=None, description="Platform to use (shopify, adobe_commerce)"
    )
    order_ids: list[str] = Field(default_factory=list, description="Known order IDs for context")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": "msg-123",
                    "conversation_id": "conv-456",
                    "customer_email": "john@example.com",
                    "text": "Where is my order #12345?",
                    "channel": "chat",
                    "platform": "shopify",
                }
            ]
        }
    }


class ConversationContext(BaseModel):
    """Context accumulated during a conversation."""

    conversation_id: str
    customer_email: str | None = None
    customer_id: str | None = None
    customer_name: str | None = None
    customer_tier: str | None = None

    # Resolved data
    order_ids: list[str] = Field(default_factory=list)
    current_order_id: str | None = None

    # Conversation history
    previous_intents: list[str] = Field(default_factory=list)
    message_count: int = 0

    # Timestamps
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_message_at: datetime | None = None


class AgentResponse(BaseModel):
    """Response from the orchestration agent."""

    message_id: str = Field(description="Original message ID")
    conversation_id: str | None = Field(default=None)

    # Intent classification results
    intents: list[dict[str, Any]] = Field(default_factory=list)
    is_compound: bool = False
    entities: list[dict[str, Any]] = Field(default_factory=list)

    # Generated response
    response_text: str = Field(description="The response to send to the customer")
    response_tone: str = Field(default="helpful", description="Tone of the response")

    # Recommended/taken actions
    actions: list[AgentAction] = Field(default_factory=list)
    primary_action: AgentAction | None = None

    # Context used
    order_context: dict[str, Any] | None = None
    customer_context: dict[str, Any] | None = None

    # Metadata
    requires_human: bool = False
    human_handoff_reason: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time_ms: int = 0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message_id": "msg-123",
                    "response_text": "Your order #12345 is currently in transit and expected to arrive by Friday.",
                    "actions": [
                        {
                            "action_type": "provide_order_status",
                            "description": "Provided order status",
                        }
                    ],
                    "confidence": 0.92,
                }
            ]
        }
    }
