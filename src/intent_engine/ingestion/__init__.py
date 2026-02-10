"""Channel ingestion adapters."""

from intent_engine.ingestion.base import ChannelAdapter
from intent_engine.ingestion.chat import ChatAdapter

__all__ = ["ChannelAdapter", "ChatAdapter"]
