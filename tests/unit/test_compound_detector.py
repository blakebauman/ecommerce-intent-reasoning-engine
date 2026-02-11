"""Tests for compound intent detection."""

import pytest

from intent_engine.matchers.compound_detector import CompoundDetector
from intent_engine.models.response import MatchResult


@pytest.fixture
def detector() -> CompoundDetector:
    """Create a compound detector instance."""
    return CompoundDetector()


class TestCompoundDetector:
    """Tests for the CompoundDetector class."""

    def test_simple_single_intent(self, detector: CompoundDetector) -> None:
        """Test that simple single intent is not flagged as compound."""
        result = detector.detect("Where is my order?")
        assert result.is_compound is False

    def test_compound_with_and(self, detector: CompoundDetector) -> None:
        """Test detection of compound with 'and also'."""
        result = detector.detect("I want to return this and also get a refund")
        assert len(result.signals) > 0

    def test_compound_multiple_sentences(self, detector: CompoundDetector) -> None:
        """Test detection of multiple action sentences."""
        result = detector.detect("I need to return this item. Also, where is my other order?")
        # Should detect multiple segments
        assert len(result.sentence_segments) >= 2

    def test_compound_with_match_hints(self, detector: CompoundDetector) -> None:
        """Test detection using match hints from different categories."""
        matches = [
            MatchResult(
                intent_code="RETURN_EXCHANGE.RETURN_INITIATE",
                similarity=0.85,
                matched_example="return this",
            ),
            MatchResult(
                intent_code="ORDER_STATUS.WISMO",
                similarity=0.80,
                matched_example="where is my order",
            ),
        ]
        result = detector.detect("Return this and where is my other order?", matches)
        # Should detect category mix
        assert any(s.signal_type == "category_mix" for s in result.signals)

    def test_get_potential_intents_single(self, detector: CompoundDetector) -> None:
        """Test getting potential intents for single intent."""
        matches = [
            MatchResult(
                intent_code="ORDER_STATUS.WISMO",
                similarity=0.90,
                matched_example="track order",
            ),
        ]
        intents = detector.get_potential_intents("Where is my order?", matches)
        assert intents == ["ORDER_STATUS.WISMO"]

    def test_sentence_segmentation(self, detector: CompoundDetector) -> None:
        """Test sentence segmentation."""
        result = detector.detect("I want to cancel my order. When will I get a refund?")
        assert len(result.sentence_segments) >= 2


class TestCompoundSignals:
    """Tests for specific compound signals."""

    def test_conjunction_detection(self, detector: CompoundDetector) -> None:
        """Test conjunction signal detection."""
        result = detector.detect("Cancel this and also track my other order")
        conjunction_signals = [s for s in result.signals if s.signal_type == "conjunction"]
        assert len(conjunction_signals) > 0

    def test_category_mix_detection(self, detector: CompoundDetector) -> None:
        """Test category mix signal detection."""
        matches = [
            MatchResult(
                intent_code="ORDER_MODIFY.CANCEL_ORDER",
                similarity=0.85,
                matched_example="cancel",
            ),
            MatchResult(
                intent_code="RETURN_EXCHANGE.REFUND_STATUS",
                similarity=0.75,
                matched_example="refund",
            ),
        ]
        result = detector.detect("Cancel and refund", matches)
        category_mix = [s for s in result.signals if s.signal_type == "category_mix"]
        assert len(category_mix) > 0
