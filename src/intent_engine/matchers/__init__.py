"""Intent matching layer - fast path classification."""

from intent_engine.matchers.compound_detector import CompoundDetector
from intent_engine.matchers.similarity import IntentMatcher, MatchDecision

__all__ = ["CompoundDetector", "IntentMatcher", "MatchDecision"]
