"""Main orchestrator for the intent reasoning engine."""

import logging
import time
from dataclasses import dataclass

from intent_engine.config import Settings, get_settings
from intent_engine.observability.metrics import record_intent_resolution, record_pipeline_stage
from intent_engine.observability.tracing import pipeline_span
from intent_engine.extractors.embedding import EmbeddingExtractor
from intent_engine.extractors.entity_extractor import EntityExtractor
from intent_engine.extractors.sentiment import SentimentAnalyzer, get_sentiment_analyzer
from intent_engine.matchers.compound_detector import CompoundDetector
from intent_engine.matchers.similarity import IntentMatcher, MatchDecision
from intent_engine.models.context import EnrichedContext
from intent_engine.models.entity import ExtractionResult
from intent_engine.models.intent import ResolvedIntent
from intent_engine.models.request import IntentRequest
from intent_engine.models.response import (
    ContextInfo,
    PolicyInfo,
    ReasoningResult,
    SentimentInfo,
)
from intent_engine.reasoners.context_enricher import ContextEnricher
from intent_engine.reasoners.decomposer import IntentDecomposer
from intent_engine.reasoners.policy_engine import PolicyEngine, get_policy_engine
from intent_engine.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class EngineComponents:
    """Container for engine components."""

    entity_extractor: EntityExtractor
    embedding_extractor: EmbeddingExtractor
    vector_store: VectorStore
    intent_matcher: IntentMatcher
    compound_detector: CompoundDetector
    decomposer: IntentDecomposer | None  # None if LLM not configured
    # Phase 2 components
    sentiment_analyzer: SentimentAnalyzer | None = None
    context_enricher: ContextEnricher | None = None
    policy_engine: PolicyEngine | None = None


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

            # Phase 2: Sentiment analyzer
            sentiment_analyzer: SentimentAnalyzer | None = None
            try:
                sentiment_analyzer = get_sentiment_analyzer()
            except Exception as e:
                logger.warning(f"Sentiment analyzer not available: {e}")

            # Phase 2: Context enricher (requires platform connector)
            context_enricher: ContextEnricher | None = None
            try:
                # Context enricher is initialized without connector by default
                # Connector can be set later via set_platform_connector()
                context_enricher = ContextEnricher()
            except Exception as e:
                logger.warning(f"Context enricher not available: {e}")

            # Phase 2: Policy engine
            policy_engine: PolicyEngine | None = None
            try:
                policy_engine = get_policy_engine()
            except Exception as e:
                logger.warning(f"Policy engine not available: {e}")

            self._components = EngineComponents(
                entity_extractor=entity_extractor,
                embedding_extractor=embedding_extractor,
                vector_store=vector_store,
                intent_matcher=intent_matcher,
                compound_detector=compound_detector,
                decomposer=decomposer,
                sentiment_analyzer=sentiment_analyzer,
                context_enricher=context_enricher,
                policy_engine=policy_engine,
            )

        self._initialized = True

    def set_platform_connector(self, connector) -> None:
        """
        Set the platform connector for context enrichment.

        Args:
            connector: A PlatformConnector instance (e.g., ShopifyConnector).
        """
        if self._components and self._components.context_enricher:
            self._components.context_enricher.connector = connector

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

        Pipeline:
        1. Entity extraction (regex + spaCy NER)
        2. Sentiment analysis (Phase 2)
        3. Embedding generation (sentence-transformers)
        4. Context enrichment (Phase 2)
        5. Similarity matching (cosine similarity)
        6. Compound intent detection
        7. Policy evaluation (Phase 2)
        8. LLM reasoning (for complex cases)
        9. Result assembly

        Args:
            request: The normalized intent request.

        Returns:
            ReasoningResult with resolved intents, sentiment, context, and policy data.
        """
        start_time = time.perf_counter()
        reasoning_trace: list[str] = []
        tenant_id = request.tenant_id

        await self.initialize()

        # Step 1: Entity extraction
        reasoning_trace.append("Step 1: Extracting entities")
        with pipeline_span("entity_extraction", tenant_id=tenant_id, request_id=request.request_id):
            stage_start = time.perf_counter()
            extraction_result = self.components.entity_extractor.extract(request.raw_text)
            record_pipeline_stage("entity_extraction", time.perf_counter() - stage_start, tenant_id)

        # Step 2: Sentiment analysis (Phase 2)
        sentiment_info: SentimentInfo | None = None
        if self.components.sentiment_analyzer:
            reasoning_trace.append("Step 2: Analyzing sentiment")
            with pipeline_span("sentiment_analysis", tenant_id=tenant_id, request_id=request.request_id):
                stage_start = time.perf_counter()
                sentiment_result = self.components.sentiment_analyzer.analyze(request.raw_text)
                record_pipeline_stage("sentiment_analysis", time.perf_counter() - stage_start, tenant_id)
            sentiment_info = SentimentInfo(
                sentiment_score=sentiment_result.sentiment_score,
                urgency_score=sentiment_result.urgency_score,
                frustration_score=sentiment_result.frustration_score,
                priority_flag=sentiment_result.priority_flag,
                signals=sentiment_result.signals,
            )
            # Update extraction result with sentiment data
            extraction_result.sentiment_score = sentiment_result.sentiment_score
            extraction_result.urgency_score = sentiment_result.urgency_score
            extraction_result.frustration_score = sentiment_result.frustration_score
            extraction_result.priority_flag = sentiment_result.priority_flag
            extraction_result.sentiment_signals = sentiment_result.signals

            if sentiment_result.priority_flag:
                reasoning_trace.append(
                    f"  Priority flag: frustration={sentiment_result.frustration_score:.2f}"
                )
        else:
            reasoning_trace.append("Step 2: Sentiment analysis (skipped - not configured)")

        # Step 3: Generate embedding
        reasoning_trace.append("Step 3: Generating embedding")
        with pipeline_span("embedding_generation", tenant_id=tenant_id, request_id=request.request_id):
            stage_start = time.perf_counter()
            embedding = self.components.embedding_extractor.embed(request.raw_text)
            extraction_result.embedding = embedding
            record_pipeline_stage("embedding_generation", time.perf_counter() - stage_start, tenant_id)

        # Step 4: Context enrichment (Phase 2)
        enriched_context: EnrichedContext | None = None
        context_info: ContextInfo | None = None
        if self.components.context_enricher:
            reasoning_trace.append("Step 4: Enriching context")
            with pipeline_span("context_enrichment", tenant_id=tenant_id, request_id=request.request_id):
                stage_start = time.perf_counter()
                try:
                    enriched_context = await self.components.context_enricher.enrich(request)
                    record_pipeline_stage("context_enrichment", time.perf_counter() - stage_start, tenant_id)
                    if enriched_context.data_sources:
                        context_info = ContextInfo(
                            customer_tier=enriched_context.customer.tier.value if enriched_context.customer else None,
                            customer_lifetime_value=enriched_context.customer.lifetime_value if enriched_context.customer else None,
                            order_status=enriched_context.order.status if enriched_context.order else None,
                            order_total=enriched_context.order.total if enriched_context.order else None,
                            is_within_return_window=enriched_context.order.is_within_return_window if enriched_context.order else None,
                            data_sources=enriched_context.data_sources,
                        )
                        reasoning_trace.append(f"  Sources: {', '.join(enriched_context.data_sources)}")
                except Exception as e:
                    logger.warning(f"Context enrichment failed: {e}")
                    reasoning_trace.append(f"  Context enrichment failed: {e}")
        else:
            reasoning_trace.append("Step 4: Context enrichment (skipped - not configured)")

        # Step 5: Similarity matching
        reasoning_trace.append("Step 5: Similarity matching")
        with pipeline_span("similarity_matching", tenant_id=tenant_id, request_id=request.request_id):
            stage_start = time.perf_counter()
            match_result = await self.components.intent_matcher.match(
                text=request.raw_text,
                embedding=embedding,
            )
            record_pipeline_stage("similarity_matching", time.perf_counter() - stage_start, tenant_id)

        reasoning_trace.append(
            f"  Top match: {match_result.top_matches[0].intent_code if match_result.top_matches else 'None'} "
            f"({match_result.top_matches[0].similarity:.2f})" if match_result.top_matches else "  No matches"
        )

        # Step 6: Check for compound intents
        reasoning_trace.append("Step 6: Checking for compound intents")
        with pipeline_span("compound_detection", tenant_id=tenant_id, request_id=request.request_id):
            stage_start = time.perf_counter()
            compound_result = self.components.compound_detector.detect(
                text=request.raw_text,
                top_matches=match_result.top_matches,
            )
            record_pipeline_stage("compound_detection", time.perf_counter() - stage_start, tenant_id)

        if compound_result.is_compound:
            reasoning_trace.append(f"  Compound signals detected: {len(compound_result.signals)}")

        # Step 7: Policy evaluation (Phase 2)
        policy_info: PolicyInfo | None = None
        primary_intent_code = ""
        if match_result.top_matches:
            primary_intent_code = match_result.top_matches[0].intent_code

        if self.components.policy_engine and enriched_context and primary_intent_code:
            reasoning_trace.append("Step 7: Evaluating policies")
            with pipeline_span("policy_evaluation", tenant_id=tenant_id, request_id=request.request_id):
                stage_start = time.perf_counter()
                try:
                    frustration = sentiment_info.frustration_score if sentiment_info else 0.0
                    policy_decision = self.components.policy_engine.evaluate(
                        context=enriched_context,
                        intent_code=primary_intent_code,
                        tenant_id=request.tenant_id,
                        frustration_score=frustration,
                    )
                    policy_info = PolicyInfo(
                        auto_approve_return=policy_decision.auto_approve_return,
                        auto_approve_refund=policy_decision.auto_approve_refund,
                        escalation_required=policy_decision.escalation_required,
                        escalation_reasons=policy_decision.escalation_reasons,
                        return_eligible=policy_decision.return_eligible,
                        return_ineligible_reason=policy_decision.return_ineligible_reason,
                        days_until_return_expires=policy_decision.days_until_return_expires,
                        recommended_action=policy_decision.recommended_action,
                        rules_applied=policy_decision.rules_applied,
                    )
                    if policy_decision.escalation_required:
                        reasoning_trace.append(
                            f"  Escalation required: {', '.join(policy_decision.escalation_reasons)}"
                        )
                    if policy_decision.auto_approve_return:
                        reasoning_trace.append("  Auto-approve return: YES")
                    record_pipeline_stage("policy_evaluation", time.perf_counter() - stage_start, tenant_id)
                except Exception as e:
                    logger.warning(f"Policy evaluation failed: {e}")
                    reasoning_trace.append(f"  Policy evaluation failed: {e}")
        else:
            reasoning_trace.append("Step 7: Policy evaluation (skipped)")

        # Decision: Fast path or reasoning path?
        if (
            match_result.decision == MatchDecision.FAST_PATH
            and not compound_result.is_compound
            and match_result.resolved_intent is not None
        ):
            # FAST PATH
            reasoning_trace.append("Decision: FAST PATH")
            processing_time = int((time.perf_counter() - start_time) * 1000)

            # Record metrics
            record_intent_resolution(
                duration_seconds=processing_time / 1000,
                tenant_id=tenant_id,
                path_taken="fast_path",
                is_compound=False,
                confidence=match_result.resolved_intent.confidence,
            )

            # Determine if human handoff needed based on policy
            requires_human = False
            human_reason = None
            if policy_info and policy_info.escalation_required:
                requires_human = True
                human_reason = f"Escalation required: {', '.join(policy_info.escalation_reasons)}"

            return ReasoningResult(
                request_id=request.request_id,
                resolved_intents=[match_result.resolved_intent],
                is_compound=False,
                entities=extraction_result.entities,
                constraints=[],
                sentiment=sentiment_info,
                policy=policy_info,
                context=context_info,
                confidence_summary=match_result.resolved_intent.confidence,
                requires_human=requires_human,
                human_handoff_reason=human_reason,
                reasoning_trace=reasoning_trace,
                processing_time_ms=processing_time,
                path_taken="fast_path",
            )

        # REASONING PATH
        reasoning_trace.append("Decision: REASONING PATH")

        # Check if LLM decomposer is available
        if self.components.decomposer is None:
            # Fallback: use best match from similarity search
            reasoning_trace.append("Step 8: LLM not available - using best match fallback")
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
                    sentiment=sentiment_info,
                    policy=policy_info,
                    context=context_info,
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
                    sentiment=sentiment_info,
                    policy=policy_info,
                    context=context_info,
                    confidence_summary=0.0,
                    requires_human=True,
                    human_handoff_reason="No matching intent found and LLM not available",
                    reasoning_trace=reasoning_trace,
                    processing_time_ms=processing_time,
                    path_taken="no_match",
                )

        reasoning_trace.append("Step 8: LLM decomposition")

        with pipeline_span("llm_decomposition", tenant_id=tenant_id, request_id=request.request_id):
            stage_start = time.perf_counter()
            decomposition = await self.components.decomposer.decompose(
            text=request.raw_text,
            entities=extraction_result.entities,
            match_hints=match_result.top_matches,
            customer_tier=request.customer_tier,
                previous_intents=request.previous_intents,
            )
            record_pipeline_stage("llm_decomposition", time.perf_counter() - stage_start, tenant_id)

        reasoning_trace.extend(decomposition.reasoning_trace)

        # Calculate overall confidence
        if decomposition.intents:
            confidence_summary = min(i.confidence for i in decomposition.intents)
        else:
            confidence_summary = 0.0

        processing_time = int((time.perf_counter() - start_time) * 1000)

        # Record metrics
        record_intent_resolution(
            duration_seconds=processing_time / 1000,
            tenant_id=tenant_id,
            path_taken="reasoning_path",
            is_compound=decomposition.is_compound,
            confidence=confidence_summary,
        )

        # Determine if human handoff needed based on policy or decomposition
        requires_human = decomposition.requires_clarification
        human_reason = decomposition.clarification_question
        if policy_info and policy_info.escalation_required:
            requires_human = True
            if human_reason:
                human_reason = f"{human_reason}; Escalation: {', '.join(policy_info.escalation_reasons)}"
            else:
                human_reason = f"Escalation required: {', '.join(policy_info.escalation_reasons)}"

        return ReasoningResult(
            request_id=request.request_id,
            resolved_intents=decomposition.intents,
            is_compound=decomposition.is_compound,
            entities=extraction_result.entities,
            constraints=decomposition.constraints,
            sentiment=sentiment_info,
            policy=policy_info,
            context=context_info,
            confidence_summary=confidence_summary,
            requires_human=requires_human,
            human_handoff_reason=human_reason,
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
