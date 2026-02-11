"""Sentiment analysis with frustration and urgency detection."""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    sentiment_score: float  # -1 (negative) to 1 (positive)
    urgency_score: float  # 0 to 1
    frustration_score: float  # 0 to 1
    priority_flag: bool  # True if should route to priority queue
    signals: list[str]  # Detected signals


class SentimentAnalyzer:
    """
    Analyze sentiment, urgency, and frustration in customer messages.

    Uses a hybrid approach:
    1. Transformer-based sentiment analysis (distilbert-sst2)
    2. Rule-based frustration and urgency detection
    3. Signal aggregation for priority routing

    Priority routing triggers when frustration_score > 0.7
    """

    # Urgency indicators
    URGENCY_PATTERNS = [
        (r"\bASAP\b", 0.9),
        (r"\burgent(ly)?\b", 0.9),
        (r"\bimmediately\b", 0.85),
        (r"\bright now\b", 0.8),
        (r"\btoday\b", 0.6),
        (r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", 0.7),
        (r"\bby (tomorrow|tonight|this morning|this afternoon|this evening)\b", 0.8),
        (r"\bwithin (\d+) (hour|day)s?\b", 0.7),
        (r"\bbefore (the weekend|monday)\b", 0.7),
        (r"\bI need (this|it) (now|today|soon)\b", 0.75),
        (r"\btime.?sensitive\b", 0.85),
        (r"\bemergency\b", 0.95),
        (r"\bcritical\b", 0.8),
        (r"\bcan't wait\b", 0.75),
        (r"\bdeadline\b", 0.7),
        (r"\bexpedite\b", 0.8),
        (r"\brush\b", 0.7),
        (r"\bpriority\b", 0.65),
        (r"\bimportant\b", 0.5),
    ]

    # Frustration indicators
    FRUSTRATION_PATTERNS = [
        # Strong frustration
        (r"\bthis is (ridiculous|unacceptable|outrageous|absurd)\b", 0.95),
        (r"\bworst (experience|service|company)\b", 0.95),
        (r"\b(horrible|terrible|awful) (experience|service)\b", 0.9),
        (r"\bI('m| am) (so )?(angry|furious|livid|fed up|done)\b", 0.9),
        (r"\bnever (ordering|buying|shopping) (here |from you )?again\b", 0.95),
        (r"\breport(ing)? (to |with )?(BBB|attorney general|lawyer)\b", 0.95),
        (r"\b(scam|fraud|steal|stolen|rip.?off)\b", 0.9),
        (r"\bcomplete(ly)? (unacceptable|incompetent)\b", 0.9),
        (r"\bwaste of (time|money)\b", 0.85),
        # Medium frustration
        (r"\bI('ve| have) (been waiting|waited|called|emailed) (\d+|\w+) times?\b", 0.75),
        (r"\bno (one|body) (is )?respond(s|ing|ed)?\b", 0.8),
        (r"\bstill (waiting|no response|nothing)\b", 0.7),
        (r"\bfor (days|weeks|months)\b", 0.65),
        (r"\bthis is the (\d+)(st|nd|rd|th) time\b", 0.8),
        (r"\b(very |really |extremely )?disappointed\b", 0.7),
        (r"\b(very |really |extremely )?frustrated\b", 0.8),
        (r"\b(very |really |extremely )?upset\b", 0.75),
        (r"\bunhappy\b", 0.6),
        (r"\bfed up\b", 0.8),
        (r"\bsick (and tired|of this)\b", 0.85),
        # Lighter frustration
        (r"\bnot (happy|satisfied|pleased)\b", 0.55),
        (r"\bfrustrat(ed|ing)\b", 0.6),
        (r"\bannoyed\b", 0.5),
        (r"\btired of\b", 0.55),
        (r"\bexhausted\b", 0.5),
        (r"\bconfused\b", 0.4),
    ]

    # Escalation phrases
    ESCALATION_PATTERNS = [
        (r"\bspeak (to|with) (a |your )?(manager|supervisor)\b", 0.8),
        (r"\bescalate\b", 0.85),
        (r"\bhigher (up|authority)\b", 0.75),
        (r"\bsomeone (else|in charge)\b", 0.65),
        (r"\breal (person|human)\b", 0.6),
        (r"\bcancel(ling|ing)? (my )?(account|subscription|membership)\b", 0.7),
        (r"\b(refund|money back)\b", 0.5),  # Contextual, lower weight
    ]

    # Negative sentiment boosters (amplify base sentiment)
    NEGATIVE_BOOSTERS = [
        r"\b(very|really|extremely|absolutely|totally|completely)\b",
        r"\b(so|such)\b",
        r"!{2,}",  # Multiple exclamation marks
        r"\bALL CAPS WORDS\b",  # Will be detected separately
    ]

    # Positive sentiment indicators (reduce frustration)
    POSITIVE_PATTERNS = [
        (r"\bthank(s| you)\b", -0.2),
        (r"\bappreciate\b", -0.2),
        (r"\bplease\b", -0.1),  # Politeness
        (r"\bgreat\b", -0.15),
        (r"\bhelp(ful|ed)\b", -0.1),
        (r"\bunderstand\b", -0.1),
    ]

    # Sarcasm indicators - positive words in negative contexts
    SARCASM_PATTERNS = [
        (r"\boh\s+great\b", 0.75),  # "Oh great, another broken item"
        (r"\bfantastic[,.]", 0.70),  # "Fantastic, it's broken"
        (r"\bwonderful[,.]", 0.70),  # "Wonderful, still waiting"
        (r"\bjust\s+perfect\b", 0.75),  # "Just perfect..."
        (r"\bjust\s+great\b", 0.75),  # "Just great..."
        (r"\bhow\s+nice\b", 0.65),  # "How nice of you to..."
        (r"\blovely[,.]", 0.60),  # "Lovely, now it's broken"
        (r"\bsuper[,.]", 0.60),  # "Super, now what"
        (r"\bawesome[,.]", 0.65),  # "Awesome, another delay"
        (r"\bgreat[,.]", 0.50),  # "Great, it's wrong" (comma after = likely sarcasm)
        (
            r"\bthanks\s+(?:so\s+much|a\s+lot)[,.]?\s+(?:for|now)",
            0.70,
        ),  # "Thanks a lot for nothing"
        (r"\breally[?!]+", 0.55),  # "Really?!" - disbelief
        (r"\bwow[,.]", 0.50),  # "Wow, still nothing"
    ]

    # Contradiction patterns (positive word + negative context in same sentence)
    CONTRADICTION_WORDS = {
        "positive": ["great", "wonderful", "fantastic", "amazing", "perfect", "awesome", "lovely"],
        "negative": [
            "broken",
            "damaged",
            "wrong",
            "missing",
            "late",
            "never",
            "still waiting",
            "another",
            "again",
            "still no",
            "don't work",
            "doesn't work",
            "failed",
        ],
    }

    PRIORITY_THRESHOLD = 0.7  # Route to priority queue above this

    def __init__(self, use_transformer: bool = True, device: str | None = None) -> None:
        """
        Initialize the sentiment analyzer.

        Args:
            use_transformer: Whether to use transformer model for base sentiment.
            device: Device to run transformer on ("cuda", "cpu", or None for auto).
        """
        self.use_transformer = use_transformer
        self.device = device
        self._pipeline = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Lazy load the transformer model."""
        if self._model_loaded:
            return

        if not self.use_transformer:
            self._model_loaded = True
            return

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
                truncation=True,
                max_length=512,
            )
            self._model_loaded = True
            logger.info("Loaded sentiment analysis model")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}. Using rule-based only.")
            self.use_transformer = False
            self._model_loaded = True

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment, urgency, and frustration in text.

        Args:
            text: The customer message to analyze.

        Returns:
            SentimentResult with all scores and signals.
        """
        self._load_model()

        signals: list[str] = []

        # 1. Get base sentiment from transformer or rules
        if self.use_transformer and self._pipeline:
            sentiment_score = self._get_transformer_sentiment(text)
            if sentiment_score < 0:
                signals.append(f"negative_sentiment:{sentiment_score:.2f}")
        else:
            sentiment_score = self._get_rule_based_sentiment(text)

        # 2. Detect urgency
        urgency_score, urgency_signals = self._detect_urgency(text)
        signals.extend(urgency_signals)

        # 3. Detect frustration
        frustration_score, frustration_signals = self._detect_frustration(text)
        signals.extend(frustration_signals)

        # 4. Check for escalation indicators
        escalation_score, escalation_signals = self._detect_escalation(text)
        signals.extend(escalation_signals)

        # 4.5. Check for sarcasm (can flip positive sentiment to negative)
        sarcasm_score, sarcasm_signals = self._detect_sarcasm(text)
        signals.extend(sarcasm_signals)

        # If sarcasm detected, flip positive sentiment to negative
        if sarcasm_score > 0.5 and sentiment_score > 0:
            sentiment_score = -sentiment_score * sarcasm_score
            signals.append(f"sentiment_flipped:{sentiment_score:.2f}")

        # 5. Boost frustration based on escalation and sarcasm
        frustration_score = min(1.0, frustration_score + (escalation_score * 0.3))
        frustration_score = min(1.0, frustration_score + (sarcasm_score * 0.2))

        # 6. Apply negative sentiment boost to frustration
        if sentiment_score < -0.3:
            frustration_boost = abs(sentiment_score) * 0.2
            frustration_score = min(1.0, frustration_score + frustration_boost)

        # 7. Check for caps lock (indicates strong emotion)
        caps_ratio = self._caps_ratio(text)
        if caps_ratio > 0.3 and len(text) > 20:
            frustration_score = min(1.0, frustration_score + 0.2)
            signals.append("excessive_caps")

        # 8. Adjust for positive indicators
        positive_adjustment = self._detect_positive(text)
        frustration_score = max(0.0, frustration_score + positive_adjustment)

        # 9. Determine priority flag
        priority_flag = frustration_score > self.PRIORITY_THRESHOLD

        if priority_flag:
            signals.append("priority_flag")

        return SentimentResult(
            sentiment_score=round(sentiment_score, 3),
            urgency_score=round(urgency_score, 3),
            frustration_score=round(frustration_score, 3),
            priority_flag=priority_flag,
            signals=signals,
        )

    def _get_transformer_sentiment(self, text: str) -> float:
        """Get sentiment from transformer model."""
        if not self._pipeline:
            return 0.0

        try:
            result = self._pipeline(text[:512])[0]
            label = result["label"]
            score = result["score"]

            # Convert to -1 to 1 scale
            if label == "NEGATIVE":
                return -score
            else:
                return score
        except Exception as e:
            logger.warning(f"Transformer inference failed: {e}")
            return 0.0

    def _get_rule_based_sentiment(self, text: str) -> float:
        """Rule-based sentiment fallback."""
        text_lower = text.lower()

        negative_count = 0
        positive_count = 0

        negative_words = [
            "bad",
            "terrible",
            "horrible",
            "awful",
            "worst",
            "hate",
            "angry",
            "disappointed",
            "frustrated",
            "upset",
            "annoyed",
            "problem",
            "issue",
            "broken",
            "damaged",
            "wrong",
            "missing",
            "late",
            "slow",
            "never",
        ]
        positive_words = [
            "good",
            "great",
            "excellent",
            "wonderful",
            "love",
            "happy",
            "satisfied",
            "thank",
            "thanks",
            "appreciate",
            "helpful",
            "perfect",
            "amazing",
        ]

        for word in negative_words:
            if word in text_lower:
                negative_count += 1

        for word in positive_words:
            if word in text_lower:
                positive_count += 1

        total = negative_count + positive_count
        if total == 0:
            return 0.0

        return (positive_count - negative_count) / total

    def _detect_urgency(self, text: str) -> tuple[float, list[str]]:
        """Detect urgency indicators."""
        text_lower = text.lower()
        max_score = 0.0
        signals: list[str] = []

        for pattern, score in self.URGENCY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                max_score = max(max_score, score)
                signals.append(f"urgency:{pattern[:20]}")

        return max_score, signals

    def _detect_frustration(self, text: str) -> tuple[float, list[str]]:
        """Detect frustration indicators."""
        text_lower = text.lower()
        scores: list[float] = []
        signals: list[str] = []

        for pattern, score in self.FRUSTRATION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                scores.append(score)
                # Extract matched text for signal
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    signals.append(f"frustration:{match.group()[:30]}")

        if not scores:
            return 0.0, signals

        # Use weighted combination: max score + average of others
        max_score = max(scores)
        if len(scores) > 1:
            other_avg = sum(scores) / len(scores)
            combined = max_score * 0.7 + other_avg * 0.3
        else:
            combined = max_score

        # Apply boosters
        booster_count = 0
        for booster in self.NEGATIVE_BOOSTERS[:-1]:  # Skip caps check
            if re.search(booster, text_lower):
                booster_count += 1

        if booster_count > 0:
            combined = min(1.0, combined + (booster_count * 0.05))

        return min(1.0, combined), signals

    def _detect_escalation(self, text: str) -> tuple[float, list[str]]:
        """Detect escalation indicators."""
        text_lower = text.lower()
        max_score = 0.0
        signals: list[str] = []

        for pattern, score in self.ESCALATION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                max_score = max(max_score, score)
                signals.append(f"escalation:{pattern[:25]}")

        return max_score, signals

    def _detect_positive(self, text: str) -> float:
        """Detect positive indicators that reduce frustration."""
        text_lower = text.lower()
        total_adjustment = 0.0

        for pattern, adjustment in self.POSITIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                total_adjustment += adjustment

        return max(-0.4, total_adjustment)  # Cap the reduction

    def _caps_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters."""
        if not text:
            return 0.0

        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0

        upper_count = sum(1 for c in alpha_chars if c.isupper())
        return upper_count / len(alpha_chars)

    def _detect_sarcasm(self, text: str) -> tuple[float, list[str]]:
        """
        Detect sarcasm in text.

        Sarcasm detection uses:
        1. Direct sarcasm patterns (e.g., "oh great", "just perfect")
        2. Contradiction detection (positive word + negative context)

        Args:
            text: The customer message to analyze.

        Returns:
            Tuple of (sarcasm score, list of sarcasm signals).
        """
        text_lower = text.lower()
        max_score = 0.0
        signals: list[str] = []

        # Check direct sarcasm patterns
        for pattern, score in self.SARCASM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                max_score = max(max_score, score)
                signals.append(f"sarcasm_pattern:{pattern[:25]}")

        # Check for contradiction (positive word near negative context)
        has_positive = any(word in text_lower for word in self.CONTRADICTION_WORDS["positive"])
        has_negative = any(phrase in text_lower for phrase in self.CONTRADICTION_WORDS["negative"])

        if has_positive and has_negative:
            # Higher confidence if they're close together
            for pos_word in self.CONTRADICTION_WORDS["positive"]:
                if pos_word in text_lower:
                    for neg_word in self.CONTRADICTION_WORDS["negative"]:
                        if neg_word in text_lower:
                            # Calculate proximity (within ~50 chars suggests same clause)
                            pos_idx = text_lower.find(pos_word)
                            neg_idx = text_lower.find(neg_word)
                            distance = abs(pos_idx - neg_idx)
                            if distance < 50:
                                contradiction_score = 0.65
                                max_score = max(max_score, contradiction_score)
                                signals.append(f"contradiction:{pos_word}+{neg_word}")
                                break

        return max_score, signals


@dataclass
class ConversationSentiment:
    """Aggregated sentiment across a conversation."""

    message_count: int
    average_frustration: float
    peak_frustration: float
    frustration_trajectory: str  # "rising", "constant", "declining"
    escalation_pattern: bool  # True if phrases like "I've asked X times"
    sentiment_history: list[float]


class ConversationSentimentTracker:
    """
    Track sentiment across conversation turns.

    Detects:
    - Escalation patterns ("I've asked 5 times")
    - Frustration trajectory (rising, constant, declining)
    - Aggregate frustration across messages
    """

    ESCALATION_PATTERNS = [
        re.compile(
            r"\b(I'?ve|I have) (asked|called|emailed|messaged|tried) (\d+|\w+) times?\b", re.I
        ),
        re.compile(r"\bthis is the (\d+)(st|nd|rd|th) time\b", re.I),
        re.compile(r"\b(again|still|yet again)\b", re.I),
        re.compile(r"\b(for (days|weeks|months))\b", re.I),
        re.compile(r"\b(multiple|several) (attempts|times|tries)\b", re.I),
    ]

    def __init__(self, analyzer: "SentimentAnalyzer | None" = None) -> None:
        """Initialize the tracker."""
        self.analyzer = analyzer or get_sentiment_analyzer()
        self._history: list[SentimentResult] = []

    def add_message(self, text: str) -> SentimentResult:
        """
        Analyze a message and add it to the conversation history.

        Args:
            text: The customer message.

        Returns:
            SentimentResult for this message.
        """
        result = self.analyzer.analyze(text)
        self._history.append(result)
        return result

    def get_conversation_sentiment(self) -> ConversationSentiment:
        """
        Get aggregated sentiment for the conversation.

        Returns:
            ConversationSentiment with trajectory and aggregation.
        """
        if not self._history:
            return ConversationSentiment(
                message_count=0,
                average_frustration=0.0,
                peak_frustration=0.0,
                frustration_trajectory="constant",
                escalation_pattern=False,
                sentiment_history=[],
            )

        frustration_scores = [r.frustration_score for r in self._history]

        # Calculate trajectory
        if len(frustration_scores) >= 3:
            first_half = sum(frustration_scores[: len(frustration_scores) // 2]) / (
                len(frustration_scores) // 2
            )
            second_half = sum(frustration_scores[len(frustration_scores) // 2 :]) / (
                len(frustration_scores) - len(frustration_scores) // 2
            )

            if second_half > first_half + 0.1:
                trajectory = "rising"
            elif second_half < first_half - 0.1:
                trajectory = "declining"
            else:
                trajectory = "constant"
        else:
            trajectory = "constant"

        # Check for escalation patterns in any message
        escalation_detected = any("escalation:" in s for r in self._history for s in r.signals)

        return ConversationSentiment(
            message_count=len(self._history),
            average_frustration=sum(frustration_scores) / len(frustration_scores),
            peak_frustration=max(frustration_scores),
            frustration_trajectory=trajectory,
            escalation_pattern=escalation_detected,
            sentiment_history=frustration_scores,
        )

    def detect_escalation_pattern(self, text: str) -> tuple[bool, list[str]]:
        """
        Detect if the message contains escalation patterns.

        Args:
            text: The customer message.

        Returns:
            Tuple of (has_escalation, signals).
        """
        signals: list[str] = []

        for pattern in self.ESCALATION_PATTERNS:
            if pattern.search(text):
                match = pattern.search(text)
                if match:
                    signals.append(f"escalation_pattern:{match.group()[:30]}")

        return len(signals) > 0, signals

    def reset(self) -> None:
        """Reset the conversation history."""
        self._history = []


# Singleton instance for easy access
_default_analyzer: SentimentAnalyzer | None = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the default sentiment analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SentimentAnalyzer()
    return _default_analyzer
