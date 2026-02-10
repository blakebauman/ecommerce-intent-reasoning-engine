"""A2A (Agent-to-Agent) protocol implementation for multi-agent orchestration."""

from intent_engine.a2a.agent_card import AgentCard, get_agent_card
from intent_engine.a2a.handler import A2ATaskHandler

__all__ = ["AgentCard", "get_agent_card", "A2ATaskHandler"]
