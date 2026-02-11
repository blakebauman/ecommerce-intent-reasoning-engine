"""Similarity-based intent matching using embedding cosine similarity."""

from dataclasses import dataclass
from enum import Enum

from intent_engine.extractors.embedding import EmbeddingExtractor
from intent_engine.models.entity import EntityType, ExtractedEntity
from intent_engine.models.intent import IntentConfidence, ResolvedIntent
from intent_engine.models.response import MatchResult
from intent_engine.storage.vector_store import VectorStore

# Mapping of intent codes to expected entity types for confidence boosting
INTENT_EXPECTED_ENTITIES: dict[str, set[EntityType]] = {
    "ORDER_STATUS.WISMO": {EntityType.ORDER_ID, EntityType.TRACKING_NUMBER},
    "ORDER_STATUS.DELIVERY_ESTIMATE": {EntityType.ORDER_ID},
    "ORDER_STATUS.TRACKING_ISSUE": {EntityType.TRACKING_NUMBER, EntityType.ORDER_ID},
    "ORDER_MODIFY.CANCEL_ORDER": {EntityType.ORDER_ID},
    "ORDER_MODIFY.CHANGE_ADDRESS": {EntityType.ORDER_ID, EntityType.ADDRESS},
    "ORDER_MODIFY.CHANGE_ITEMS": {
        EntityType.ORDER_ID,
        EntityType.PRODUCT_SKU,
        EntityType.SIZE,
        EntityType.COLOR,
    },
    "RETURN_EXCHANGE.RETURN_INITIATE": {EntityType.ORDER_ID, EntityType.REASON},
    "RETURN_EXCHANGE.EXCHANGE_REQUEST": {EntityType.ORDER_ID, EntityType.SIZE, EntityType.COLOR},
    "RETURN_EXCHANGE.REFUND_STATUS": {EntityType.ORDER_ID, EntityType.MONEY_AMOUNT},
    "COMPLAINT.DAMAGED_ITEM": {EntityType.ORDER_ID, EntityType.REASON},
    "COMPLAINT.WRONG_ITEM": {
        EntityType.ORDER_ID,
        EntityType.PRODUCT_SKU,
        EntityType.COLOR,
        EntityType.SIZE,
    },
    "COMPLAINT.MISSING_ITEM": {EntityType.ORDER_ID, EntityType.PRODUCT_SKU},
    "PRODUCT_INQUIRY.STOCK": {EntityType.PRODUCT_SKU, EntityType.SIZE, EntityType.COLOR},
    "PRODUCT_INQUIRY.COMPATIBILITY": {EntityType.PRODUCT_SKU},
}

# Confidence boost when extracted entities match expected entities
ENTITY_CONFIDENCE_BOOST = 0.05


class MatchDecision(str, Enum):
    """Decision from the matching layer."""

    FAST_PATH = "fast_path"  # High confidence, single intent → resolve immediately
    REASONING_PATH = "reasoning_path"  # Needs LLM reasoning
    CLARIFICATION = "clarification"  # Too ambiguous, need to ask user


@dataclass
class MatchingResult:
    """Result from intent matching."""

    decision: MatchDecision
    top_matches: list[MatchResult]
    resolved_intent: ResolvedIntent | None = None
    is_ambiguous: bool = False
    ambiguity_reason: str | None = None


class IntentMatcher:
    """
    Fast path intent matching using embedding similarity.

    Uses cosine similarity against the intent catalog to classify
    unambiguous single intents. Routes complex cases to the reasoning path.

    Thresholds:
    - >= 0.85: HIGH confidence → fast path (if no ambiguity)
    - 0.60-0.85: MEDIUM confidence → reasoning path
    - < 0.60: LOW confidence → reasoning path (may need clarification)

    Ambiguity check: If second-best match is within 0.10 of top match,
    route to reasoning path even if top match is high confidence.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_extractor: EmbeddingExtractor | None = None,
        fast_path_threshold: float = 0.85,
        ambiguity_gap_threshold: float = 0.10,
        low_confidence_threshold: float = 0.60,
    ) -> None:
        """
        Initialize the intent matcher.

        Args:
            vector_store: Vector store for similarity search.
            embedding_extractor: Embedding generator.
            fast_path_threshold: Minimum similarity for fast path (default 0.85).
            ambiguity_gap_threshold: Min gap between top-2 matches (default 0.10).
            low_confidence_threshold: Below this, may need clarification (default 0.60).
        """
        self.vector_store = vector_store
        self.embedding_extractor = embedding_extractor or EmbeddingExtractor()
        self.fast_path_threshold = fast_path_threshold
        self.ambiguity_gap_threshold = ambiguity_gap_threshold
        self.low_confidence_threshold = low_confidence_threshold

    async def match(
        self,
        text: str,
        embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> MatchingResult:
        """
        Match text against the intent catalog.

        Args:
            text: The customer message to classify.
            embedding: Pre-computed embedding (optional, computed if not provided).
            top_k: Number of matches to retrieve.

        Returns:
            MatchingResult with decision and matches.
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding_extractor.embed(text)

        # Search for similar intents
        matches = await self.vector_store.similarity_search(
            embedding=embedding,
            top_k=top_k,
        )

        if not matches:
            return MatchingResult(
                decision=MatchDecision.CLARIFICATION,
                top_matches=[],
                is_ambiguous=True,
                ambiguity_reason="No matches found in intent catalog",
            )

        # Convert to MatchResult objects
        top_matches = [
            MatchResult(
                intent_code=m.intent_code,
                similarity=m.similarity,
                matched_example=m.example_text,
            )
            for m in matches
        ]

        top_match = matches[0]

        # Check for low confidence
        if top_match.similarity < self.low_confidence_threshold:
            return MatchingResult(
                decision=MatchDecision.REASONING_PATH,
                top_matches=top_matches,
                is_ambiguous=True,
                ambiguity_reason=f"Low confidence ({top_match.similarity:.2f})",
            )

        # Check for ambiguity (second match too close)
        if len(matches) > 1:
            second_match = matches[1]
            gap = top_match.similarity - second_match.similarity

            if gap < self.ambiguity_gap_threshold:
                # Different categories = definitely ambiguous
                if top_match.category != second_match.category:
                    return MatchingResult(
                        decision=MatchDecision.REASONING_PATH,
                        top_matches=top_matches,
                        is_ambiguous=True,
                        ambiguity_reason=(
                            f"Multiple categories: {top_match.intent_code} vs "
                            f"{second_match.intent_code} (gap: {gap:.2f})"
                        ),
                    )
                # Same category but different intents
                elif top_match.intent_code != second_match.intent_code:
                    return MatchingResult(
                        decision=MatchDecision.REASONING_PATH,
                        top_matches=top_matches,
                        is_ambiguous=True,
                        ambiguity_reason=(
                            f"Close match: {second_match.intent_code} "
                            f"({second_match.similarity:.2f})"
                        ),
                    )

        # Check for high confidence fast path
        if top_match.similarity >= self.fast_path_threshold:
            # Parse intent code
            parts = top_match.intent_code.split(".")
            category = parts[0]
            intent = parts[1] if len(parts) > 1 else parts[0]

            resolved = ResolvedIntent(
                category=category,
                intent=intent,
                confidence=top_match.similarity,
                confidence_tier=IntentConfidence.HIGH,
                evidence=[top_match.example_text],
            )

            return MatchingResult(
                decision=MatchDecision.FAST_PATH,
                top_matches=top_matches,
                resolved_intent=resolved,
            )

        # Medium confidence → reasoning path
        return MatchingResult(
            decision=MatchDecision.REASONING_PATH,
            top_matches=top_matches,
            is_ambiguous=False,
            ambiguity_reason=None,
        )

    async def match_with_hints(
        self,
        text: str,
        embedding: list[float] | None = None,
    ) -> tuple[list[MatchResult], list[str]]:
        """
        Get top matches as hints for the LLM reasoning path.

        Args:
            text: The customer message.
            embedding: Pre-computed embedding (optional).

        Returns:
            Tuple of (top matches, list of intent codes as hints).
        """
        result = await self.match(text, embedding, top_k=3)
        hints = [m.intent_code for m in result.top_matches]
        return result.top_matches, hints

    async def match_with_entity_boost(
        self,
        text: str,
        entities: list[ExtractedEntity],
        embedding: list[float] | None = None,
        top_k: int = 5,
    ) -> MatchingResult:
        """
        Match with entity-based confidence boosting.

        When extracted entities overlap with expected entity types for an intent,
        apply a 5% confidence boost to the match similarity.

        Args:
            text: The customer message to classify.
            entities: Extracted entities from the message.
            embedding: Pre-computed embedding (optional).
            top_k: Number of matches to retrieve.

        Returns:
            MatchingResult with potentially boosted confidence.
        """
        # Get base match result
        result = await self.match(text, embedding, top_k)

        if not result.top_matches or not entities:
            return result

        # Extract entity types present in the message
        extracted_types = {e.entity_type for e in entities}

        # Check for entity overlap with top match
        top_match = result.top_matches[0]
        expected_entities = INTENT_EXPECTED_ENTITIES.get(top_match.intent_code, set())

        if expected_entities and extracted_types.intersection(expected_entities):
            # Apply confidence boost
            boosted_similarity = min(1.0, top_match.similarity * (1 + ENTITY_CONFIDENCE_BOOST))

            # Update the top match with boosted similarity
            result.top_matches[0] = MatchResult(
                intent_code=top_match.intent_code,
                similarity=boosted_similarity,
                matched_example=top_match.matched_example,
            )

            # If this pushes us over the fast path threshold, update the result
            if (
                result.decision != MatchDecision.FAST_PATH
                and boosted_similarity >= self.fast_path_threshold
                and not result.is_ambiguous
            ):
                parts = top_match.intent_code.split(".")
                category = parts[0]
                intent = parts[1] if len(parts) > 1 else parts[0]

                overlapping = extracted_types.intersection(expected_entities)
                evidence = [
                    top_match.matched_example,
                    f"Entity boost: {', '.join(e.value for e in overlapping)}",
                ]

                result.resolved_intent = ResolvedIntent(
                    category=category,
                    intent=intent,
                    confidence=boosted_similarity,
                    confidence_tier=IntentConfidence.HIGH,
                    evidence=evidence,
                )
                result.decision = MatchDecision.FAST_PATH

        return result
