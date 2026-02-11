"""Unit tests for IntentEngine with mocked EngineComponents."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Skip on Python 3.14: spacy/sentence_transformers not compatible; run with just test-docker
pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="spacy/sentence_transformers not compatible with Python 3.14; use just test-docker",
)

if sys.version_info < (3, 14):
    from intent_engine.engine import EngineComponents, IntentEngine
    from intent_engine.matchers.similarity import MatchDecision, MatchingResult
    from intent_engine.models.entity import EntityType, ExtractedEntity, ExtractionResult
    from intent_engine.models.intent import IntentConfidence, ResolvedIntent
    from intent_engine.models.request import InputChannel, IntentRequest
    from intent_engine.models.response import MatchResult
    from intent_engine.reasoners.decomposer import DecompositionOutput
else:
    # Dummies so collection succeeds when skipped
    EngineComponents = IntentEngine = MatchDecision = MatchingResult = None
    ExtractionResult = ExtractedEntity = EntityType = None
    IntentConfidence = ResolvedIntent = InputChannel = IntentRequest = None
    MatchResult = DecompositionOutput = None


def _make_components(
    *,
    match_decision: MatchDecision | None = None,
    top_similarity: float = 0.92,
    is_compound: bool = False,
    decomposer: object | None = None,
    top_matches: list | None = None,
) -> EngineComponents:
    """Build EngineComponents with mocks for engine.resolve() tests."""
    if match_decision is None:
        match_decision = MatchDecision.FAST_PATH
    extraction_result = ExtractionResult(
        entities=[
            ExtractedEntity(
                entity_type=EntityType.ORDER_ID,
                value="12345",
                raw_span="#12345",
                start_pos=0,
                end_pos=5,
                confidence=0.99,
            )
        ],
        embedding=[0.1] * 384,
    )

    entity_extractor = MagicMock()
    entity_extractor.extract.return_value = extraction_result

    embedding_extractor = MagicMock()
    embedding_extractor.embed.return_value = [0.1] * 384

    vector_store = MagicMock()

    if top_matches is None:
        top_matches = [
            MatchResult(
                intent_code="ORDER_STATUS.WISMO",
                similarity=top_similarity,
                matched_example="where is my order",
            )
        ]
    resolved_intent = (
        ResolvedIntent(
            category="ORDER_STATUS",
            intent="WISMO",
            confidence=top_similarity,
            confidence_tier=IntentConfidence.HIGH,
            evidence=["where is my order"],
        )
        if match_decision == MatchDecision.FAST_PATH and top_matches
        else None
    )

    match_result = MatchingResult(
        decision=match_decision,
        top_matches=top_matches,
        resolved_intent=resolved_intent,
    )

    intent_matcher = MagicMock()
    intent_matcher.match = AsyncMock(return_value=match_result)

    compound_result = MagicMock()
    compound_result.is_compound = is_compound
    compound_result.signals = []

    compound_detector = MagicMock()
    compound_detector.detect.return_value = compound_result

    return EngineComponents(
        entity_extractor=entity_extractor,
        embedding_extractor=embedding_extractor,
        vector_store=vector_store,
        intent_matcher=intent_matcher,
        compound_detector=compound_detector,
        decomposer=decomposer,
        sentiment_analyzer=None,
        context_enricher=None,
        policy_engine=None,
        conflict_resolver=MagicMock(),
    )


@pytest.fixture
def sample_request() -> IntentRequest:
    """Minimal intent request for tests."""
    return IntentRequest(
        request_id="test-req",
        tenant_id="test-tenant",
        channel=InputChannel.CHAT,
        raw_text="Where is my order #12345?",
    )


@pytest.mark.asyncio
async def test_resolve_fast_path(sample_request: IntentRequest) -> None:
    """Engine returns fast path result when match is high confidence and not compound."""
    components = _make_components(
        match_decision=MatchDecision.FAST_PATH,
        top_similarity=0.92,
        is_compound=False,
    )
    engine = IntentEngine(components=components)
    await engine.initialize()

    result = await engine.resolve(sample_request)

    assert result.path_taken == "fast_path"
    assert len(result.resolved_intents) == 1
    assert result.resolved_intents[0].category == "ORDER_STATUS"
    assert result.resolved_intents[0].intent == "WISMO"
    assert result.is_compound is False
    assert result.entities
    assert result.entities[0].entity_type == EntityType.ORDER_ID
    assert result.entities[0].value == "12345"


@pytest.mark.asyncio
async def test_resolve_fast_path_fallback_when_no_decomposer(
    sample_request: IntentRequest,
) -> None:
    """When reasoning path is needed but decomposer is None, engine uses best match fallback."""
    components = _make_components(
        match_decision=MatchDecision.REASONING_PATH,
        top_similarity=0.75,
        is_compound=False,
        decomposer=None,
    )
    engine = IntentEngine(components=components)
    await engine.initialize()

    result = await engine.resolve(sample_request)

    assert result.path_taken == "fast_path_fallback"
    assert len(result.resolved_intents) == 1
    assert result.resolved_intents[0].category == "ORDER_STATUS"
    assert result.resolved_intents[0].intent == "WISMO"
    assert result.requires_human is True
    assert "LLM" in (result.human_handoff_reason or "")


@pytest.mark.asyncio
async def test_resolve_reasoning_path_with_decomposer(
    sample_request: IntentRequest,
) -> None:
    """When reasoning path and decomposer is set, engine uses LLM decomposition."""
    decomposition = DecompositionOutput(
        intents=[
            ResolvedIntent(
                category="ORDER_STATUS",
                intent="WISMO",
                confidence=0.88,
                confidence_tier=IntentConfidence.HIGH,
                evidence=["decomposed"],
            )
        ],
        is_compound=False,
        constraints=[],
        requires_clarification=False,
        clarification_question=None,
        reasoning_trace=["Step 1: Decomposed"],
    )
    mock_decomposer = MagicMock()
    mock_decomposer.decompose = AsyncMock(return_value=decomposition)

    components = _make_components(
        match_decision=MatchDecision.REASONING_PATH,
        top_similarity=0.72,
        is_compound=False,
        decomposer=mock_decomposer,
    )
    engine = IntentEngine(components=components)
    await engine.initialize()

    result = await engine.resolve(sample_request)

    assert result.path_taken == "reasoning_path"
    assert len(result.resolved_intents) == 1
    assert result.resolved_intents[0].intent == "WISMO"
    mock_decomposer.decompose.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_compound_detected_uses_reasoning_path(
    sample_request: IntentRequest,
) -> None:
    """Compound intent detection forces reasoning path even with high similarity."""
    components = _make_components(
        match_decision=MatchDecision.FAST_PATH,
        top_similarity=0.90,
        is_compound=True,  # compound signal
        decomposer=None,
    )
    engine = IntentEngine(components=components)
    await engine.initialize()

    result = await engine.resolve(sample_request)

    # Should go to reasoning path; with no decomposer we get fallback
    assert result.path_taken == "fast_path_fallback"
    assert result.is_compound is True


@pytest.mark.asyncio
async def test_resolve_text_convenience_method() -> None:
    """resolve_text() builds IntentRequest and returns same shape as resolve()."""
    components = _make_components(
        match_decision=MatchDecision.FAST_PATH,
        top_similarity=0.91,
        is_compound=False,
    )
    engine = IntentEngine(components=components)
    await engine.initialize()

    result = await engine.resolve_text(
        "Where is my order?",
        request_id="inline-1",
        tenant_id="default",
    )

    assert result.path_taken == "fast_path"
    assert result.request_id == "inline-1"
    assert result.resolved_intents[0].intent == "WISMO"


@pytest.mark.asyncio
async def test_resolve_no_match_when_empty_matches_and_no_decomposer(
    sample_request: IntentRequest,
) -> None:
    """When matcher returns no matches and no decomposer, engine returns no_match result."""
    components = _make_components(
        match_decision=MatchDecision.REASONING_PATH,
        top_similarity=0.0,
        is_compound=False,
        decomposer=None,
        top_matches=[],  # No catalog matches
    )
    engine = IntentEngine(components=components)
    await engine.initialize()

    result = await engine.resolve(sample_request)

    assert result.path_taken == "no_match"
    assert len(result.resolved_intents) == 0
    assert result.requires_human is True
    assert "No matching intent" in (result.human_handoff_reason or "")
