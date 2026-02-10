"""Base channel adapter interface."""

from abc import ABC, abstractmethod
from typing import Any

from intent_engine.models.request import IntentRequest


class ChannelAdapter(ABC):
    """
    Abstract base class for channel adapters.

    Each channel (chat, email, form, etc.) implements this interface
    to normalize its input format into a unified IntentRequest.
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Return the channel name (e.g., 'chat', 'email')."""
        ...

    @abstractmethod
    async def normalize(self, raw_input: dict[str, Any]) -> IntentRequest:
        """
        Convert channel-specific payload to IntentRequest.

        Args:
            raw_input: The raw input from the channel.

        Returns:
            Normalized IntentRequest.
        """
        ...

    @abstractmethod
    def validate(self, raw_input: dict[str, Any]) -> bool:
        """
        Validate channel-specific payload structure.

        Args:
            raw_input: The raw input to validate.

        Returns:
            True if valid, False otherwise.
        """
        ...

    def extract_order_ids(self, text: str) -> list[str]:
        """
        Extract order IDs from text (helper method).

        Can be overridden by subclasses for channel-specific patterns.
        """
        import re

        patterns = [
            r"#?\b(ORD[-_]?\d{4,10})\b",
            r"#?\b(ORDER[-_]?\d{4,10})\b",
            r"#(\d{4,10})\b",
        ]

        order_ids: list[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            order_ids.extend(matches)

        return list(set(order_ids))
