"""Main orchestrator for the intent reasoning engine."""

import time
from dataclasses import dataclass

from intent_engine.config import Settings, get_settings
from intent_engine.extractors.embedding import EmbeddingExtractor
from intent_engine.extractors.entity_extractor import EntityExtractor
from intent_engine.matchers.compound_detector import CompoundDetector
from intent_engine.matchers.similarity import IntentMatcher, MatchDecision
from intent_engine.models.entity import ExtractionResult
from intent_engine.models.intent import ResolvedIntent
from intent_engine.models.request import IntentRequest
from intent_engine.models.response import ReasoningResult
from intent_engine.reasoners.decomposer import IntentDecomposer
from intent_engine.storage.vector_store import VectorStore


@dataclass
class EngineComponents:
    """Container for engine components."""

    entity_extractor: EntityExtractor
    embedding_extractor: EmbeddingExtractor
    vector_store: VectorStore
    intent_matcher: IntentMatcher
    compound_detector: CompoundDetector
    decomposer: IntentDecomposer | None  # None if LLM not configured


class IntentEngine:
    """
    Main orchestrator for the intent reasoning engine.

    Coordinates the full pipeline:
    1. Entity extraction (regex + spaCy NER)
    2. Embedding generation (sentence-transformers)
    3. Fast path matching (cosine similarity)
    4. Compound intent detection
    5. LLM reasoning (for complex cases)
    6. Result assembly
    """

    def __init__(
        self,
        settings: Settings | None = None,
        components: EngineComponents | None = None,
    ) -> None:
        """
        Initialize the intent engine.

        Args:
            settings: Configuration settings. Uses defaults if not provided.
            components: Pre-configured components (for testing).
        """
        self.settings = settings or get_settings()
        self._components = components
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize engine components and connections."""
        if self._initialized:
            return

        if self._components is None:
            # Create components
            entity_extractor = EntityExtractor(
                spacy_model=self.settings.spacy_model
            )
            embedding_extractor = EmbeddingExtractor(
                model_name=self.settings.embedding_model
            )
            vector_store = VectorStore(self.settings.database_url)
            await vector_store.connect()

            intent_matcher = IntentMatcher(
                vector_store=vector_store,
                embedding_extractor=embedding_extractor,
                fast_path_threshold=self.settings.fast_path_threshold,
                ambiguity_gap_threshold=self.settings.ambiguity_gap_threshold,
                low_confidence_threshold=self.settings.low_confidence_threshold,
            )
            compound_detector = CompoundDetector(
                compound_threshold=self.settings.compound_detection_threshold
            )

            # Decomposer is optional - requires ANTHROPIC_API_KEY
            decomposer: IntentDecomposer | None = None
            try:
                decomposer = IntentDecomposer(
                    model_name=self.settings.llm_model
                )
            except Exception as e:
                import logging
                logging.warning(
                    f"LLM decomposer not available: {e}. "
                    "Fast path only - set ANTHROPIC_API_KEY for full functionality."
                )

            self._components = EngineComponents(
                entity_extractor=entity_extractor,
                embedding_extractor=embedding_extractor,
                vector_store=vector_store,
                intent_matcher=intent_matcher,
                compound_detector=compound_detector,
                decomposer=decomposer,
            )

        self._initialized = True

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._components:
            await self._components.vector_store.close()
        self._initialized = False

    @property
    def components(self) -> EngineComponents:
        """Get engine components."""
        if not self._components:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        return self._components

    async def resolve(self, request: IntentRequest) -> ReasoningResult:
        """
        Resolve intents from a customer request.

        This is the main entry point for intent classification.

        Args:
            request: The normalized intent request.

        Returns:
            ReasoningResult with resolved intents and metadata.
        """
        start_time = time.perf_counter()
        reasoning_trace: list[str] = []

        await self.initialize()

        # Step 1: Entity extraction
        reasoning_trace.append("Step 1: Extracting entities")
        extraction_result = self.components.entity_extractor.extract(request.raw_text)

        # Step 2: Generate embedding
        reasoning_trace.append("Step 2: Generating embedding")
        embedding = self.components.embedding_extractor.embed(request.raw_text)
        extraction_result.embedding = embedding

        # Step 3: Similarity matching
        reasoning_trace.append("Step 3: Similarity matching")
        match_result = await self.components.intent_matcher.match(
            text=request.raw_text,
            embedding=embedding,
        )

        reasoning_trace.append(
            f"  Top match: {match_result.top_matches[0].intent_code if match_result.top_matches else 'None'} "
            f"({match_result.top_matches[0].similarity:.2f})" if match_result.top_matches else "  No matches"
        )

        # Step 4: Check for compound intents
        reasoning_trace.append("Step 4: Checking for compound intents")
        compound_result = self.components.compound_detector.detect(
            text=request.raw_text,
            top_matches=match_result.top_matches,
        )

        if compound_result.is_compound:
            reasoning_trace.append(f"  Compound signals detected: {len(compound_result.signals)}")

        # Decision: Fast path or reasoning path?
        if (
            match_result.decision == MatchDecision.FAST_PATH
            and not compound_result.is_compound
            and match_result.resolved_intent is not None
        ):
            # FAST PATH
            reasoning_trace.append("Decision: FAST PATH")
            processing_time = int((time.perf_counter() - start_time) * 1000)

            return ReasoningResult(
                request_id=request.request_id,
                resolved_intents=[match_result.resolved_intent],
                is_compound=False,
                entities=extraction_result.entities,
                constraints=[],
                confidence_summary=match_result.resolved_intent.confidence,
                requires_human=False,
                reasoning_trace=reasoning_trace,
                processing_time_ms=processing_time,
                path_taken="fast_path",
            )

        # REASONING PATH
        reasoning_trace.append("Decision: REASONING PATH")

        # Check if LLM decomposer is available
        if self.components.decomposer is None:
            # Fallback: use best match from similarity search
            reasoning_trace.append("Step 5: LLM not available - using best match fallback")
            processing_time = int((time.perf_counter() - start_time) * 1000)

            if match_result.top_matches:
                best_match = match_result.top_matches[0]
                # Parse intent code (e.g., "ORDER_STATUS.WISMO" -> category="ORDER_STATUS", intent="WISMO")
                parts = best_match.intent_code.split(".")
                category = parts[0] if parts else "UNKNOWN"
                intent_name = parts[1] if len(parts) > 1 else "UNKNOWN"

                from intent_engine.models.intent import IntentConfidence
                if best_match.similarity >= 0.85:
                    tier = IntentConfidence.HIGH
                elif best_match.similarity >= 0.60:
                    tier = IntentConfidence.MEDIUM
                else:
                    tier = IntentConfidence.LOW

                fallback_intent = ResolvedIntent(
                    category=category,
                    intent=intent_name,
                    confidence=best_match.similarity,
                    confidence_tier=tier,
                    evidence=[f"Best match (fallback): {best_match.matched_example[:50]}..."],
                )
                return ReasoningResult(
                    request_id=request.request_id,
                    resolved_intents=[fallback_intent],
                    is_compound=compound_result.is_compound,
                    entities=extraction_result.entities,
                    constraints=[],
                    confidence_summary=best_match.similarity,
                    requires_human=True,
                    human_handoff_reason="LLM reasoning not available - low confidence match",
                    reasoning_trace=reasoning_trace,
                    processing_time_ms=processing_time,
                    path_taken="fast_path_fallback",
                )
            else:
                return ReasoningResult(
                    request_id=request.request_id,
                    resolved_intents=[],
                    is_compound=False,
                    entities=extraction_result.entities,
                    constraints=[],
                    confidence_summary=0.0,
                    requires_human=True,
                    human_handoff_reason="No matching intent found and LLM not available",
                    reasoning_trace=reasoning_trace,
                    processing_time_ms=processing_time,
                    path_taken="no_match",
                )

        reasoning_trace.append("Step 5: LLM decomposition")

        decomposition = await self.components.decomposer.decompose(
            text=request.raw_text,
            entities=extraction_result.entities,
            match_hints=match_result.top_matches,
            customer_tier=request.customer_tier,
            previous_intents=request.previous_intents,
        )

        reasoning_trace.extend(decomposition.reasoning_trace)

        # Calculate overall confidence
        if decomposition.intents:
            confidence_summary = min(i.confidence for i in decomposition.intents)
        else:
            confidence_summary = 0.0

        processing_time = int((time.perf_counter() - start_time) * 1000)

        return ReasoningResult(
            request_id=request.request_id,
            resolved_intents=decomposition.intents,
            is_compound=decomposition.is_compound,
            entities=extraction_result.entities,
            constraints=decomposition.constraints,
            confidence_summary=confidence_summary,
            requires_human=decomposition.requires_clarification,
            human_handoff_reason=decomposition.clarification_question,
            reasoning_trace=reasoning_trace,
            processing_time_ms=processing_time,
            path_taken="reasoning_path",
        )

    async def resolve_text(
        self,
        text: str,
        request_id: str = "inline",
        tenant_id: str = "default",
    ) -> ReasoningResult:
        """
        Convenience method to resolve a raw text string.

        Args:
            text: The customer message.
            request_id: Optional request ID.
            tenant_id: Optional tenant ID.

        Returns:
            ReasoningResult with resolved intents.
        """
        from intent_engine.models.request import InputChannel, IntentRequest

        request = IntentRequest(
            request_id=request_id,
            tenant_id=tenant_id,
            channel=InputChannel.CHAT,
            raw_text=text,
        )

        return await self.resolve(request)
