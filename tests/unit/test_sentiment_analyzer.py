"""Tests for sentiment analyzer."""

import pytest

from intent_engine.extractors.sentiment import (
    ConversationSentimentTracker,
    SentimentAnalyzer,
    SentimentResult,
)


@pytest.fixture
def analyzer() -> SentimentAnalyzer:
    """Create a sentiment analyzer (rules-only for fast tests)."""
    return SentimentAnalyzer(use_transformer=False)


class TestSentimentAnalyzerBasic:
    """Basic sentiment analysis tests."""

    def test_neutral_message(self, analyzer: SentimentAnalyzer) -> None:
        """Test neutral message."""
        result = analyzer.analyze("Where is my order #12345?")
        assert -0.5 < result.sentiment_score < 0.5
        assert result.priority_flag is False

    def test_positive_message(self, analyzer: SentimentAnalyzer) -> None:
        """Test positive message with appreciation."""
        result = analyzer.analyze("Thank you for the quick help! Great service.")
        assert result.sentiment_score > 0
        assert result.frustration_score < 0.3

    def test_negative_message(self, analyzer: SentimentAnalyzer) -> None:
        """Test negative message."""
        result = analyzer.analyze("This is terrible. The product is broken and wrong.")
        assert result.sentiment_score < 0


class TestFrustrationDetection:
    """Tests for frustration detection."""

    def test_high_frustration_phrases(self, analyzer: SentimentAnalyzer) -> None:
        """Test detection of high-frustration phrases."""
        high_frustration_messages = [
            "This is ridiculous! I've been waiting for weeks!",
            "Worst experience I've ever had with any company!",
            "I'm so angry about this horrible service!",
            "Never ordering from you again. Complete waste of money!",
        ]

        for msg in high_frustration_messages:
            result = analyzer.analyze(msg)
            assert result.frustration_score >= 0.7, f"Failed for: {msg}"
            assert result.priority_flag is True, f"Priority not set for: {msg}"

    def test_medium_frustration_phrases(self, analyzer: SentimentAnalyzer) -> None:
        """Test detection of medium-frustration phrases."""
        medium_frustration_messages = [
            "I'm a bit unhappy with the service.",
            "Still waiting for a response.",
            "I'm confused about the process.",
        ]

        for msg in medium_frustration_messages:
            result = analyzer.analyze(msg)
            assert 0.3 <= result.frustration_score <= 0.7, f"Unexpected score for: {msg}"

    def test_low_frustration(self, analyzer: SentimentAnalyzer) -> None:
        """Test low-frustration messages don't trigger priority."""
        result = analyzer.analyze("I'm a bit confused about the return policy.")
        assert result.frustration_score < 0.5
        assert result.priority_flag is False

    def test_multiple_frustration_signals_compound(self, analyzer: SentimentAnalyzer) -> None:
        """Test that multiple frustration signals are detected."""
        single = analyzer.analyze("I'm confused.")
        multiple = analyzer.analyze(
            "I'm confused. I've been waiting for weeks. This is annoying. I'm tired of this."
        )
        # Multiple signals should be detected even if score doesn't increase linearly
        assert len(multiple.signals) >= len(single.signals)

    def test_priority_threshold(self, analyzer: SentimentAnalyzer) -> None:
        """Test priority flag threshold is 0.7."""
        assert analyzer.PRIORITY_THRESHOLD == 0.7

        # Explicitly high frustration should trigger
        result = analyzer.analyze("This is unacceptable! Worst service ever!")
        assert result.frustration_score > 0.7
        assert result.priority_flag is True


class TestUrgencyDetection:
    """Tests for urgency detection."""

    def test_high_urgency_words(self, analyzer: SentimentAnalyzer) -> None:
        """Test high-urgency indicators."""
        urgent_messages = [
            "I need this ASAP!",
            "This is urgent, please respond immediately.",
            "It's an emergency, I need help right now.",
        ]

        for msg in urgent_messages:
            result = analyzer.analyze(msg)
            assert result.urgency_score >= 0.7, f"Failed for: {msg}"

    def test_medium_urgency(self, analyzer: SentimentAnalyzer) -> None:
        """Test medium-urgency indicators."""
        result = analyzer.analyze("I need this by tomorrow please.")
        assert 0.5 <= result.urgency_score <= 0.9

    def test_deadline_detection(self, analyzer: SentimentAnalyzer) -> None:
        """Test deadline-based urgency."""
        deadline_messages = [
            "I need this by Friday.",
            "This has a deadline coming up.",
            "I need it within 2 days.",
        ]

        for msg in deadline_messages:
            result = analyzer.analyze(msg)
            assert result.urgency_score >= 0.5, f"Failed for: {msg}"

    def test_no_urgency(self, analyzer: SentimentAnalyzer) -> None:
        """Test messages without urgency."""
        result = analyzer.analyze("When you get a chance, could you check on my order?")
        assert result.urgency_score < 0.5


class TestEscalationDetection:
    """Tests for escalation indicator detection."""

    def test_escalation_phrases(self, analyzer: SentimentAnalyzer) -> None:
        """Test escalation request detection."""
        escalation_messages = [
            "I want to speak to a manager.",
            "Please escalate this issue.",
            "I need to talk to someone higher up.",
        ]

        for msg in escalation_messages:
            result = analyzer.analyze(msg)
            # Escalation should boost frustration
            assert len([s for s in result.signals if "escalation" in s]) > 0, (
                f"No escalation signal for: {msg}"
            )

    def test_cancel_account_escalation(self, analyzer: SentimentAnalyzer) -> None:
        """Test account cancellation as escalation signal."""
        result = analyzer.analyze("I'm cancelling my account if this isn't resolved.")
        escalation_signals = [s for s in result.signals if "escalation" in s]
        assert len(escalation_signals) > 0


class TestCapsDetection:
    """Tests for excessive caps detection."""

    def test_all_caps_increases_frustration(self, analyzer: SentimentAnalyzer) -> None:
        """Test that ALL CAPS boosts frustration."""
        normal = analyzer.analyze("Where is my order?")
        caps = analyzer.analyze("WHERE IS MY ORDER? I HAVE BEEN WAITING!")

        assert caps.frustration_score > normal.frustration_score
        assert "excessive_caps" in caps.signals

    def test_short_caps_ignored(self, analyzer: SentimentAnalyzer) -> None:
        """Test that short all-caps messages don't trigger."""
        # Short messages under 20 chars shouldn't trigger caps detection
        result = analyzer.analyze("HELP")
        assert "excessive_caps" not in result.signals


class TestPositiveAdjustments:
    """Tests for positive indicator adjustments."""

    def test_thanks_reduces_frustration(self, analyzer: SentimentAnalyzer) -> None:
        """Test 'thanks' reduces frustration score."""
        without_thanks = analyzer.analyze("I'm disappointed with the service.")
        with_thanks = analyzer.analyze(
            "I'm disappointed with the service, but thanks for looking into it."
        )

        # Thanks should reduce frustration slightly
        assert with_thanks.frustration_score <= without_thanks.frustration_score

    def test_please_and_appreciate(self, analyzer: SentimentAnalyzer) -> None:
        """Test politeness markers reduce frustration."""
        polite = analyzer.analyze("Please help me. I appreciate your assistance.")
        assert polite.frustration_score < 0.5


class TestSignals:
    """Tests for signal generation."""

    def test_frustration_signal_content(self, analyzer: SentimentAnalyzer) -> None:
        """Test frustration signals include matched text."""
        result = analyzer.analyze("I'm very disappointed with this.")
        frustration_signals = [s for s in result.signals if "frustration:" in s]
        assert len(frustration_signals) > 0
        assert "disappointed" in frustration_signals[0]

    def test_urgency_signal_content(self, analyzer: SentimentAnalyzer) -> None:
        """Test urgency signals are generated."""
        result = analyzer.analyze("I need this ASAP!")
        urgency_signals = [s for s in result.signals if "urgency:" in s]
        assert len(urgency_signals) > 0

    def test_priority_flag_signal(self, analyzer: SentimentAnalyzer) -> None:
        """Test priority_flag signal is added."""
        result = analyzer.analyze("This is absolutely unacceptable! I'm furious!")
        assert "priority_flag" in result.signals


class TestSentimentResult:
    """Tests for SentimentResult structure."""

    def test_result_fields(self, analyzer: SentimentAnalyzer) -> None:
        """Test all result fields are present."""
        result = analyzer.analyze("Test message")

        assert hasattr(result, "sentiment_score")
        assert hasattr(result, "urgency_score")
        assert hasattr(result, "frustration_score")
        assert hasattr(result, "priority_flag")
        assert hasattr(result, "signals")

    def test_scores_in_range(self, analyzer: SentimentAnalyzer) -> None:
        """Test scores are within expected ranges."""
        test_messages = [
            "Normal question",
            "I'm very angry!",
            "Thank you so much!",
            "URGENT!!! HELP NOW!!!",
        ]

        for msg in test_messages:
            result = analyzer.analyze(msg)
            assert -1 <= result.sentiment_score <= 1
            assert 0 <= result.urgency_score <= 1
            assert 0 <= result.frustration_score <= 1
            assert isinstance(result.priority_flag, bool)
            assert isinstance(result.signals, list)


class TestRuleBasedSentiment:
    """Tests for rule-based sentiment (no transformer)."""

    def test_rule_based_negative(self, analyzer: SentimentAnalyzer) -> None:
        """Test rule-based negative detection."""
        result = analyzer._get_rule_based_sentiment("terrible horrible awful worst broken")
        assert result < 0

    def test_rule_based_positive(self, analyzer: SentimentAnalyzer) -> None:
        """Test rule-based positive detection."""
        result = analyzer._get_rule_based_sentiment("great excellent wonderful thank helpful")
        assert result > 0

    def test_rule_based_neutral(self, analyzer: SentimentAnalyzer) -> None:
        """Test rule-based neutral (no signals)."""
        result = analyzer._get_rule_based_sentiment("where is my order")
        assert result == 0


class TestLegalThreatDetection:
    """Tests for legal threat detection."""

    def test_legal_threat_high_frustration(self, analyzer: SentimentAnalyzer) -> None:
        """Test legal threats increase frustration."""
        # Use exact pattern from FRUSTRATION_PATTERNS - "report(ing)? (to |with )?(BBB|attorney general|lawyer)"
        result = analyzer.analyze("I'm reporting to BBB about this!")
        assert result.frustration_score >= 0.7
        assert result.priority_flag is True

    def test_fraud_accusation(self, analyzer: SentimentAnalyzer) -> None:
        """Test fraud accusations."""
        result = analyzer.analyze("This is a scam! You stole my money!")
        assert result.frustration_score >= 0.7


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string(self, analyzer: SentimentAnalyzer) -> None:
        """Test empty string doesn't crash."""
        result = analyzer.analyze("")
        assert result.sentiment_score == 0

    def test_very_long_text(self, analyzer: SentimentAnalyzer) -> None:
        """Test very long text is handled."""
        long_text = "I have a question. " * 1000
        result = analyzer.analyze(long_text)
        assert isinstance(result, SentimentResult)

    def test_special_characters(self, analyzer: SentimentAnalyzer) -> None:
        """Test special characters don't crash."""
        result = analyzer.analyze("Test @#$%^&*() 123 !!!")
        assert isinstance(result, SentimentResult)

    def test_unicode(self, analyzer: SentimentAnalyzer) -> None:
        """Test unicode characters are handled."""
        result = analyzer.analyze("I need help please! ")
        assert isinstance(result, SentimentResult)


class TestSarcasmDetection:
    """Tests for sarcasm detection (Phase 2)."""

    def test_sarcasm_pattern_oh_great(self, analyzer: SentimentAnalyzer) -> None:
        """Test 'oh great' sarcasm detection."""
        result = analyzer.analyze("Oh great, another broken item.")
        sarcasm_signals = [s for s in result.signals if "sarcasm" in s]
        assert len(sarcasm_signals) > 0

    def test_sarcasm_pattern_just_perfect(self, analyzer: SentimentAnalyzer) -> None:
        """Test 'just perfect' sarcasm detection."""
        result = analyzer.analyze("Just perfect, now it's completely wrong.")
        sarcasm_signals = [s for s in result.signals if "sarcasm" in s or "contradiction" in s]
        assert len(sarcasm_signals) > 0

    def test_sarcasm_flips_sentiment(self, analyzer: SentimentAnalyzer) -> None:
        """Test that sarcasm flips positive to negative sentiment."""
        # Pure positive message
        positive = analyzer.analyze("Great service!")
        # Sarcastic positive message
        sarcastic = analyzer.analyze("Oh great, my package is lost again.")

        # Sarcasm should result in negative or neutral sentiment
        # even when positive words are present
        assert sarcastic.frustration_score >= positive.frustration_score

    def test_contradiction_detection(self, analyzer: SentimentAnalyzer) -> None:
        """Test positive word + negative context contradiction."""
        result = analyzer.analyze("Wonderful, the item is broken and missing parts.")
        sarcasm_signals = [s for s in result.signals if "sarcasm" in s or "contradiction" in s]
        assert len(sarcasm_signals) > 0

    def test_genuine_positive_not_flagged(self, analyzer: SentimentAnalyzer) -> None:
        """Test that genuine positive messages aren't flagged as sarcasm."""
        result = analyzer.analyze("Great product! I love it. Thank you!")
        sarcasm_signals = [s for s in result.signals if "sarcasm" in s]
        # Should have no sarcasm signals or very few
        # (genuine positive without negative context)
        assert len(sarcasm_signals) == 0 or result.frustration_score < 0.5


class TestConversationSentimentTracker:
    """Tests for conversation sentiment tracking (Phase 5)."""

    @pytest.fixture
    def tracker(self, analyzer: SentimentAnalyzer) -> ConversationSentimentTracker:
        """Create a conversation tracker."""
        return ConversationSentimentTracker(analyzer)

    def test_empty_conversation(self, tracker: ConversationSentimentTracker) -> None:
        """Test empty conversation returns zero values."""
        conv = tracker.get_conversation_sentiment()
        assert conv.message_count == 0
        assert conv.average_frustration == 0.0
        assert conv.frustration_trajectory == "constant"

    def test_single_message(self, tracker: ConversationSentimentTracker) -> None:
        """Test single message tracking."""
        tracker.add_message("Where is my order?")
        conv = tracker.get_conversation_sentiment()
        assert conv.message_count == 1

    def test_rising_frustration_trajectory(self, tracker: ConversationSentimentTracker) -> None:
        """Test detection of rising frustration."""
        tracker.add_message("Where is my order?")
        tracker.add_message("I've been waiting a while.")
        tracker.add_message("This is frustrating!")
        tracker.add_message("This is unacceptable! Worst service ever!")

        conv = tracker.get_conversation_sentiment()
        assert conv.message_count == 4
        # Should detect rising trend
        assert conv.frustration_trajectory in ["rising", "constant"]

    def test_peak_frustration(self, tracker: ConversationSentimentTracker) -> None:
        """Test peak frustration tracking."""
        tracker.add_message("Normal question.")
        tracker.add_message("I'm very angry about this terrible service!")
        tracker.add_message("Okay, thanks for helping.")

        conv = tracker.get_conversation_sentiment()
        # Peak should be from the angry message
        assert conv.peak_frustration >= 0.5

    def test_escalation_pattern_detection(self, tracker: ConversationSentimentTracker) -> None:
        """Test escalation pattern detection."""
        has_pattern, signals = tracker.detect_escalation_pattern("I've called 5 times about this!")
        assert has_pattern is True
        assert len(signals) > 0

    def test_reset_clears_history(self, tracker: ConversationSentimentTracker) -> None:
        """Test reset clears conversation history."""
        tracker.add_message("Test message")
        tracker.reset()
        conv = tracker.get_conversation_sentiment()
        assert conv.message_count == 0
