"""Entity extraction using regex patterns and spaCy NER."""

import re
from typing import Any

import dateparser
import spacy
from spacy.language import Language

from intent_engine.models.entity import EntityType, ExtractedEntity, ExtractionResult


class EntityExtractor:
    """
    Extract eCommerce entities from customer messages.

    Uses a combination of:
    - Regex patterns for structured identifiers (order IDs, tracking numbers, etc.)
    - spaCy NER for general entities (names, addresses, money)
    - dateparser for date/deadline extraction
    """

    # Regex patterns for eCommerce entities
    PATTERNS: dict[EntityType, list[re.Pattern[str]]] = {
        EntityType.ORDER_ID: [
            # Common order ID formats: #12345, ORD-12345, ORDER-12345, etc.
            re.compile(r"#?\b(ORD[-_]?\d{4,10})\b", re.IGNORECASE),
            re.compile(r"#?\b(ORDER[-_]?\d{4,10})\b", re.IGNORECASE),
            re.compile(r"\border(?:er)?\s*(?:number|#|id)?[:\s]*#?(\d{4,10})\b", re.IGNORECASE),
            re.compile(r"#(\d{4,10})\b"),  # Simple # followed by digits
        ],
        EntityType.TRACKING_NUMBER: [
            # USPS: 20-22 digits or 13 chars starting with letters
            re.compile(r"\b(\d{20,22})\b"),
            re.compile(r"\b([A-Z]{2}\d{9}[A-Z]{2})\b"),
            # UPS: 1Z followed by 16 alphanumeric
            re.compile(r"\b(1Z[A-Z0-9]{16})\b", re.IGNORECASE),
            # FedEx: 12-15 digits
            re.compile(r"\btracking[:\s#]*(\d{12,15})\b", re.IGNORECASE),
        ],
        EntityType.PRODUCT_SKU: [
            re.compile(r"\bsku[:\s#]*([A-Z0-9]{4,12})\b", re.IGNORECASE),
            re.compile(r"\bitem[:\s#]*([A-Z0-9]{4,12})\b", re.IGNORECASE),
        ],
        EntityType.SIZE: [
            re.compile(r"\bsize[:\s]*(XXS|XS|S|M|L|XL|XXL|XXXL|\d{1,2})\b", re.IGNORECASE),
            re.compile(r"\b(small|medium|large|extra\s*large)\b", re.IGNORECASE),
        ],
        EntityType.COLOR: [
            re.compile(
                r"\b(red|blue|green|yellow|black|white|pink|purple|orange|"
                r"brown|gray|grey|navy|beige|tan|gold|silver)\b",
                re.IGNORECASE,
            ),
        ],
        EntityType.QUANTITY: [
            re.compile(r"\b(\d{1,3})\s*(?:items?|pieces?|units?|qty)\b", re.IGNORECASE),
            re.compile(r"\bqty[:\s]*(\d{1,3})\b", re.IGNORECASE),
        ],
        EntityType.EMAIL: [
            re.compile(r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"),
        ],
        EntityType.PHONE: [
            re.compile(r"\b(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"),
        ],
    }

    # Deadline keywords and patterns
    DEADLINE_PATTERNS = [
        re.compile(r"\bby\s+(friday|saturday|sunday|monday|tuesday|wednesday|thursday)\b", re.I),
        re.compile(r"\bwithin\s+(\d+)\s*(days?|hours?|weeks?)\b", re.IGNORECASE),
        re.compile(r"\bbefore\s+(\w+\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b", re.I),
        re.compile(r"\bneed(?:ed)?\s+(?:it\s+)?by\s+(\w+)\b", re.IGNORECASE),
        re.compile(r"\b(urgent|asap|immediately|rush)\b", re.IGNORECASE),
    ]

    # Return/complaint reason keywords
    REASON_PATTERNS = [
        re.compile(r"\b(damaged|broken|defective|wrong|incorrect|missing|late)\b", re.IGNORECASE),
        re.compile(r"\b(doesn't fit|too small|too large|wrong size|wrong color)\b", re.IGNORECASE),
        re.compile(r"\b(changed my mind|no longer need|found cheaper|duplicate order)\b", re.I),
        re.compile(r"\b(not as described|fake|counterfeit|poor quality)\b", re.IGNORECASE),
    ]

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize the entity extractor with a spaCy model."""
        self._nlp: Language | None = None
        self._spacy_model = spacy_model

    @property
    def nlp(self) -> Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = spacy.load(self._spacy_model)
        return self._nlp

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all entities from the input text.

        Args:
            text: The customer message to extract entities from.

        Returns:
            ExtractionResult containing all extracted entities.
        """
        entities: list[ExtractedEntity] = []

        # Extract using regex patterns
        entities.extend(self._extract_regex_entities(text))

        # Extract using spaCy NER
        entities.extend(self._extract_spacy_entities(text))

        # Extract dates and deadlines
        entities.extend(self._extract_dates(text))

        # Extract reasons
        entities.extend(self._extract_reasons(text))

        # Deduplicate entities (prefer higher confidence)
        entities = self._deduplicate_entities(entities)

        return ExtractionResult(entities=entities)

    def _extract_regex_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities: list[ExtractedEntity] = []

        for entity_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Use the first capturing group if present, else whole match
                    value = match.group(1) if match.lastindex else match.group(0)
                    entities.append(
                        ExtractedEntity(
                            entity_type=entity_type,
                            value=value.strip(),
                            raw_span=match.group(0),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=0.95,  # High confidence for regex matches
                        )
                    )

        return entities

    def _extract_spacy_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        entities: list[ExtractedEntity] = []
        doc = self.nlp(text)

        entity_mapping: dict[str, EntityType] = {
            "PERSON": EntityType.PERSON_NAME,
            "GPE": EntityType.ADDRESS,  # Geopolitical entity (cities, countries)
            "LOC": EntityType.ADDRESS,
            "MONEY": EntityType.MONEY_AMOUNT,
            "CARDINAL": EntityType.QUANTITY,
            "PRODUCT": EntityType.PRODUCT_NAME,
            "ORG": EntityType.PRODUCT_NAME,  # Sometimes product/brand names
        }

        for ent in doc.ents:
            if ent.label_ in entity_mapping:
                entities.append(
                    ExtractedEntity(
                        entity_type=entity_mapping[ent.label_],
                        value=ent.text,
                        raw_span=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=0.80,  # Moderate confidence for NER
                    )
                )

        return entities

    def _extract_dates(self, text: str) -> list[ExtractedEntity]:
        """Extract dates and deadlines from text."""
        entities: list[ExtractedEntity] = []

        # Check for deadline patterns
        for pattern in self.DEADLINE_PATTERNS:
            for match in pattern.finditer(text):
                raw_span = match.group(0)
                value = match.group(1) if match.lastindex else match.group(0)

                # Try to parse into an actual date
                parsed_date = dateparser.parse(
                    value,
                    settings={"PREFER_DATES_FROM": "future"},
                )

                entities.append(
                    ExtractedEntity(
                        entity_type=EntityType.DEADLINE,
                        value=parsed_date.isoformat() if parsed_date else value,
                        raw_span=raw_span,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.85 if parsed_date else 0.70,
                    )
                )

        return entities

    def _extract_reasons(self, text: str) -> list[ExtractedEntity]:
        """Extract return/complaint reasons from text."""
        entities: list[ExtractedEntity] = []

        for pattern in self.REASON_PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    ExtractedEntity(
                        entity_type=EntityType.REASON,
                        value=match.group(0).lower(),
                        raw_span=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.90,
                    )
                )

        return entities

    def _deduplicate_entities(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Remove duplicate entities, keeping the highest confidence one."""
        seen: dict[tuple[EntityType, str], ExtractedEntity] = {}

        for entity in entities:
            key = (entity.entity_type, entity.value.lower())
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def extract_order_ids(self, text: str) -> list[str]:
        """Convenience method to extract just order IDs."""
        result = self.extract(text)
        return [
            e.value
            for e in result.entities
            if e.entity_type == EntityType.ORDER_ID
        ]
