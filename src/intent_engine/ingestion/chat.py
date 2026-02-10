"""Chat channel adapter."""

import uuid
from datetime import datetime
from typing import Any

from intent_engine.ingestion.base import ChannelAdapter
from intent_engine.models.request import InputChannel, IntentRequest


class ChatAdapter(ChannelAdapter):
    """
    Adapter for chat/messaging channel input.

    Normalizes chat messages from various chat platforms (Intercom,
    Zendesk, custom widgets, etc.) into IntentRequest format.

    Expected input format:
    {
        "message": "Where is my order?",
        "session_id": "sess-123",
        "tenant_id": "merchant-1",
        "customer_email": "user@example.com",
        "customer_id": "cust-456",
        "timestamp": "2024-02-09T10:30:00Z",  # optional
        "metadata": {  # optional
            "agent_id": "agent-1",
            "widget_context": "order_page",
            "previous_messages": [...]
        }
    }
    """

    @property
    def channel_name(self) -> str:
        return "chat"

    def validate(self, raw_input: dict[str, Any]) -> bool:
        """Validate chat input structure."""
        required_fields = ["message", "tenant_id"]

        for field in required_fields:
            if field not in raw_input or not raw_input[field]:
                return False

        # Message must be a non-empty string
        if not isinstance(raw_input["message"], str):
            return False

        if len(raw_input["message"].strip()) == 0:
            return False

        return True

    async def normalize(self, raw_input: dict[str, Any]) -> IntentRequest:
        """
        Normalize chat input to IntentRequest.

        Args:
            raw_input: Raw chat message data.

        Returns:
            Normalized IntentRequest.
        """
        if not self.validate(raw_input):
            raise ValueError("Invalid chat input: missing required fields")

        message = raw_input["message"]
        tenant_id = raw_input["tenant_id"]

        # Generate request ID if not provided
        request_id = raw_input.get("request_id", str(uuid.uuid4()))

        # Parse timestamp
        timestamp = datetime.utcnow()
        if ts := raw_input.get("timestamp"):
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    pass
            elif isinstance(ts, datetime):
                timestamp = ts

        # Extract metadata
        metadata = raw_input.get("metadata", {})

        # Build raw_metadata with chat-specific fields
        raw_metadata: dict[str, Any] = {
            "session_id": raw_input.get("session_id"),
            "agent_id": metadata.get("agent_id"),
            "widget_context": metadata.get("widget_context"),
        }

        # Extract previous messages for context
        previous_messages = metadata.get("previous_messages", [])
        if previous_messages:
            raw_metadata["previous_messages"] = previous_messages[-5:]  # Last 5 messages

        # Extract order IDs from message
        order_ids = self.extract_order_ids(message)

        # Extract previous intents if available
        previous_intents: list[str] = []
        if "previous_intents" in raw_input:
            previous_intents = raw_input["previous_intents"]
        elif "resolved_intents" in metadata:
            previous_intents = metadata["resolved_intents"]

        # Determine message index (position in conversation)
        message_index = 0
        if previous_messages:
            message_index = len(previous_messages)

        return IntentRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            channel=InputChannel.CHAT,
            timestamp=timestamp,
            raw_text=message,
            raw_metadata=raw_metadata,
            conversation_id=raw_input.get("session_id") or raw_input.get("conversation_id"),
            message_index=message_index,
            previous_intents=previous_intents,
            customer_id=raw_input.get("customer_id"),
            customer_tier=raw_input.get("customer_tier"),
            order_ids=order_ids + raw_input.get("order_ids", []),
            page_context=metadata.get("widget_context"),
        )

    def build_context_from_history(
        self,
        current_message: str,
        previous_messages: list[dict[str, Any]],
        max_messages: int = 5,
    ) -> str:
        """
        Build context string from conversation history.

        Useful for multi-turn conversations where context matters.

        Args:
            current_message: The current customer message.
            previous_messages: List of previous messages in the conversation.
            max_messages: Maximum number of previous messages to include.

        Returns:
            Context string with conversation history.
        """
        if not previous_messages:
            return current_message

        # Take last N messages
        recent = previous_messages[-max_messages:]

        # Build context string
        context_parts: list[str] = []
        for msg in recent:
            role = msg.get("role", "customer")
            text = msg.get("text", msg.get("message", ""))
            if text:
                context_parts.append(f"[{role}]: {text}")

        # Add current message
        context_parts.append(f"[customer]: {current_message}")

        return "\n".join(context_parts)
