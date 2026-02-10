"""Intent decomposition using Pydantic AI agent."""

from dataclasses import dataclass

from pydantic_ai import Agent

from intent_engine.llm.client import (
    DecompositionResult,
    IntentContext,
    get_intent_agent,
)
from intent_engine.models.entity import ExtractedEntity
from intent_engine.models.intent import IntentConfidence, ResolvedIntent
from intent_engine.models.response import Constraint, MatchResult


@dataclass
class DecompositionOutput:
    """Output from the intent decomposer."""

    intents: list[ResolvedIntent]
    is_compound: bool
    constraints: list[Constraint]
    requires_clarification: bool
    clarification_question: str | None
    reasoning_trace: list[str]


class IntentDecomposer:
    """
    Decompose compound or ambiguous intents using LLM reasoning.

    Uses Pydantic AI with Claude to:
    - Break compound requests into atomic intents
    - Resolve ambiguous classifications
    - Extract constraints (deadlines, preferences)
    - Generate clarification questions when needed
    - Use tools for order lookup and return eligibility checks
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5",
        agent: Agent[IntentContext, DecompositionResult] | None = None,
        order_lookup: any = None,
        return_eligibility_check: any = None,
    ) -> None:
        """
        Initialize the decomposer.

        Args:
            model_name: Anthropic model name.
            agent: Optional pre-configured agent (for testing).
            order_lookup: Optional async callback to look up order details.
            return_eligibility_check: Optional async callback to check return eligibility.
        """
        self.agent = agent or get_intent_agent(model_name)
        self._model_name = model_name
        self._order_lookup = order_lookup
        self._return_eligibility_check = return_eligibility_check

    async def decompose(
        self,
        text: str,
        entities: list[ExtractedEntity] | None = None,
        match_hints: list[MatchResult] | None = None,
        customer_tier: str | None = None,
        previous_intents: list[str] | None = None,
    ) -> DecompositionOutput:
        """
        Decompose a customer message into atomic intents.

        Args:
            text: The customer message.
            entities: Extracted entities from the message.
            match_hints: Top similarity matches as hints.
            customer_tier: Customer tier (VIP, standard, etc.).
            previous_intents: Intents from earlier in the conversation.

        Returns:
            DecompositionOutput with resolved intents and metadata.
        """
        # Build context for the agent
        entity_dicts = [
            {
                "entity_type": e.entity_type.value,
                "value": e.value,
                "confidence": e.confidence,
            }
            for e in (entities or [])
        ]

        hint_codes = [m.intent_code for m in (match_hints or [])]

        context = IntentContext(
            raw_text=text,
            extracted_entities=entity_dicts,
            match_hints=hint_codes,
            customer_tier=customer_tier,
            previous_intents=previous_intents,
            order_lookup=self._order_lookup,
            return_eligibility_check=self._return_eligibility_check,
        )

        # Run the agent
        result = await self.agent.run(text, deps=context)
        decomposition = result.output

        # Build reasoning trace
        trace = [
            f"Reasoning path: LLM decomposition ({self._model_name})",
            f"Input: '{text[:100]}...' " if len(text) > 100 else f"Input: '{text}'",
        ]

        if hint_codes:
            trace.append(f"Match hints: {hint_codes}")

        trace.append(f"LLM reasoning: {decomposition.reasoning}")

        # Convert to ResolvedIntent objects
        resolved_intents: list[ResolvedIntent] = []
        for intent in decomposition.intents:
            # Determine confidence tier
            if intent.confidence >= 0.85:
                tier = IntentConfidence.HIGH
            elif intent.confidence >= 0.60:
                tier = IntentConfidence.MEDIUM
            else:
                tier = IntentConfidence.LOW

            # Parse intent code
            parts = intent.intent_code.split(".")
            category = parts[0]
            intent_name = parts[1] if len(parts) > 1 else parts[0]

            resolved_intents.append(
                ResolvedIntent(
                    category=category,
                    intent=intent_name,
                    confidence=intent.confidence,
                    confidence_tier=tier,
                    evidence=intent.evidence,
                )
            )

        # Extract constraints
        constraints: list[Constraint] = []
        for intent in decomposition.intents:
            for constraint_str in intent.constraints:
                # Parse constraint type from string
                constraint_type = "preference"
                if any(w in constraint_str.lower() for w in ["by", "before", "deadline"]):
                    constraint_type = "deadline"
                elif any(w in constraint_str.lower() for w in ["must", "require", "need"]):
                    constraint_type = "requirement"

                constraints.append(
                    Constraint(
                        constraint_type=constraint_type,
                        description=constraint_str,
                        value=constraint_str,
                        hard=constraint_type == "deadline",
                    )
                )

        if resolved_intents:
            intent_summary = ", ".join(i.intent_code for i in resolved_intents)
            trace.append(f"Resolved intents: {intent_summary}")

        if decomposition.is_compound:
            trace.append(f"Compound intent detected: {len(resolved_intents)} atomic intents")

        return DecompositionOutput(
            intents=resolved_intents,
            is_compound=decomposition.is_compound,
            constraints=constraints,
            requires_clarification=decomposition.requires_clarification,
            clarification_question=decomposition.clarification_question,
            reasoning_trace=trace,
        )
