"""Compound intent detection - identifies multi-intent requests."""

import re
from dataclasses import dataclass

from intent_engine.models.response import MatchResult


@dataclass
class CompoundSignal:
    """A signal indicating potential compound intent."""

    signal_type: str  # "conjunction", "multiple_sentences", "category_mix"
    description: str
    confidence: float


@dataclass
class CompoundDetectionResult:
    """Result from compound intent detection."""

    is_compound: bool
    signals: list[CompoundSignal]
    sentence_segments: list[str]
    confidence: float  # Overall confidence that this is compound


class CompoundDetector:
    """
    Detect compound intents in customer messages.

    Identifies multi-intent requests using:
    - Linguistic signals: "and", "also", "but", "then", etc.
    - Multiple sentences with different action verbs
    - Top-k matches spanning multiple intent categories

    Any compound signal routes to the reasoning path for LLM decomposition.
    """

    # Conjunctions that often indicate compound intents
    COMPOUND_CONJUNCTIONS = [
        r"\band\s+(?:also|then|I\s+also)\b",
        r"\bbut\s+also\b",
        r"\balso\s+(?:want|need|would\s+like)\b",
        r"\bplus\b",
        r"\bas\s+well\s+as\b",
        r"\bin\s+addition\b",
        r"\bon\s+top\s+of\s+that\b",
        r"\bwhile\s+you'?re\s+at\s+it\b",
    ]

    # Action verbs that indicate intent
    ACTION_VERBS = [
        r"\b(cancel|return|exchange|refund|track|replace|change|update|modify)\b",
        r"\b(where\s+is|when\s+will|how\s+do\s+I|can\s+I|I\s+want|I\s+need)\b",
        r"\b(check|find|get|send|ship|deliver)\b",
    ]

    def __init__(
        self,
        compound_threshold: float = 0.60,
    ) -> None:
        """
        Initialize the compound detector.

        Args:
            compound_threshold: Minimum confidence to flag as compound.
        """
        self.compound_threshold = compound_threshold
        self._conjunction_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.COMPOUND_CONJUNCTIONS
        ]
        self._action_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.ACTION_VERBS
        ]

    def detect(
        self,
        text: str,
        top_matches: list[MatchResult] | None = None,
    ) -> CompoundDetectionResult:
        """
        Detect if the text contains compound intents.

        Args:
            text: The customer message to analyze.
            top_matches: Optional similarity matches to check for category mixing.

        Returns:
            CompoundDetectionResult with signals and confidence.
        """
        signals: list[CompoundSignal] = []

        # Check for compound conjunctions
        conjunction_signals = self._detect_conjunctions(text)
        signals.extend(conjunction_signals)

        # Check for multiple sentences with different actions
        sentences = self._segment_sentences(text)
        sentence_signals = self._detect_multi_action_sentences(sentences)
        signals.extend(sentence_signals)

        # Check for category mixing in top matches
        if top_matches:
            category_signals = self._detect_category_mix(top_matches)
            signals.extend(category_signals)

        # Calculate overall confidence
        if not signals:
            confidence = 0.0
        else:
            # Weight signals and combine
            confidence = min(1.0, sum(s.confidence for s in signals) / 2)

        is_compound = confidence >= self.compound_threshold

        return CompoundDetectionResult(
            is_compound=is_compound,
            signals=signals,
            sentence_segments=sentences,
            confidence=confidence,
        )

    def _detect_conjunctions(self, text: str) -> list[CompoundSignal]:
        """Detect compound conjunctions in text."""
        signals: list[CompoundSignal] = []

        for pattern in self._conjunction_patterns:
            if pattern.search(text):
                signals.append(
                    CompoundSignal(
                        signal_type="conjunction",
                        description=f"Found compound conjunction: {pattern.pattern}",
                        confidence=0.70,
                    )
                )

        return signals

    def _segment_sentences(self, text: str) -> list[str]:
        """Split text into sentence segments."""
        # Split on sentence boundaries and common separators
        segments = re.split(r"[.!?]\s+|\s*[,;]\s+(?=and|but|also|I\s)", text)
        # Clean up and filter empty segments
        return [s.strip() for s in segments if s.strip() and len(s.strip()) > 3]

    def _detect_multi_action_sentences(
        self, sentences: list[str]
    ) -> list[CompoundSignal]:
        """Detect multiple sentences with different action verbs."""
        signals: list[CompoundSignal] = []

        if len(sentences) < 2:
            return signals

        actions_per_sentence: list[set[str]] = []

        for sentence in sentences:
            actions: set[str] = set()
            for pattern in self._action_patterns:
                matches = pattern.findall(sentence.lower())
                actions.update(matches)
            if actions:
                actions_per_sentence.append(actions)

        # Check if we have multiple sentences with different actions
        if len(actions_per_sentence) >= 2:
            all_actions = set().union(*actions_per_sentence)
            if len(all_actions) >= 2:
                signals.append(
                    CompoundSignal(
                        signal_type="multiple_sentences",
                        description=(
                            f"Found {len(actions_per_sentence)} segments with "
                            f"different actions: {all_actions}"
                        ),
                        confidence=0.80,
                    )
                )

        return signals

    def _detect_category_mix(
        self, top_matches: list[MatchResult]
    ) -> list[CompoundSignal]:
        """Detect mixing of intent categories in top matches."""
        signals: list[CompoundSignal] = []

        if len(top_matches) < 2:
            return signals

        # Get categories from top 3 matches with decent similarity
        categories: set[str] = set()
        for match in top_matches[:3]:
            if match.similarity >= 0.50:
                category = match.intent_code.split(".")[0]
                categories.add(category)

        if len(categories) >= 2:
            signals.append(
                CompoundSignal(
                    signal_type="category_mix",
                    description=f"Top matches span categories: {categories}",
                    confidence=0.75,
                )
            )

        return signals

    def get_potential_intents(
        self,
        text: str,
        top_matches: list[MatchResult],
    ) -> list[str]:
        """
        Get list of potential intent codes for compound decomposition.

        Args:
            text: The customer message.
            top_matches: Similarity matches.

        Returns:
            List of intent codes that might be present.
        """
        result = self.detect(text, top_matches)

        if not result.is_compound:
            # Just return the top match
            return [top_matches[0].intent_code] if top_matches else []

        # For compound, return top matches with decent similarity
        return [
            m.intent_code
            for m in top_matches[:3]
            if m.similarity >= 0.50
        ]
