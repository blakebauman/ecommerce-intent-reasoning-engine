"""Intent catalog management."""

import json
from pathlib import Path
from typing import Any

from intent_engine.extractors.embedding import EmbeddingExtractor
from intent_engine.models.intent import CoreIntent
from intent_engine.storage.vector_store import VectorStore


class IntentCatalogStore:
    """
    High-level intent catalog management.

    Handles loading examples from JSON files, generating embeddings,
    and populating the vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_extractor: EmbeddingExtractor | None = None,
    ) -> None:
        """
        Initialize the catalog store.

        Args:
            vector_store: The underlying vector storage.
            embedding_extractor: Embedding generator (created if not provided).
        """
        self.vector_store = vector_store
        self.embedding_extractor = embedding_extractor or EmbeddingExtractor()

    async def load_from_json(self, filepath: str | Path) -> dict[str, int]:
        """
        Load intent examples from a JSON file and populate the catalog.

        Expected JSON format:
        {
            "ORDER_STATUS.WISMO": [
                "Where is my order?",
                "Track my package",
                ...
            ],
            "ORDER_STATUS.DELIVERY_ESTIMATE": [...],
            ...
        }

        Args:
            filepath: Path to the JSON file.

        Returns:
            Dict mapping intent codes to number of examples loaded.
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data: dict[str, list[str]] = json.load(f)

        counts: dict[str, int] = {}

        for intent_code, examples in data.items():
            # Validate intent code
            category = intent_code.split(".")[0]

            # Generate embeddings for all examples
            embeddings = self.embedding_extractor.embed_batch(examples)

            # Prepare records for batch insert
            records = [
                (intent_code, category, example, embedding)
                for example, embedding in zip(examples, embeddings)
            ]

            # Insert into vector store
            await self.vector_store.insert_embeddings_batch(records)
            counts[intent_code] = len(examples)

        return counts

    async def add_examples(
        self,
        intent_code: str,
        examples: list[str],
    ) -> int:
        """
        Add new examples for an intent.

        Args:
            intent_code: The intent code (e.g., "ORDER_STATUS.WISMO").
            examples: List of example utterances.

        Returns:
            Number of examples added.
        """
        category = intent_code.split(".")[0]
        embeddings = self.embedding_extractor.embed_batch(examples)

        records = [
            (intent_code, category, example, embedding)
            for example, embedding in zip(examples, embeddings)
        ]

        return await self.vector_store.insert_embeddings_batch(records)

    async def get_catalog_stats(self) -> dict[str, Any]:
        """
        Get statistics about the intent catalog.

        Returns:
            Dict with catalog statistics.
        """
        counts = await self.vector_store.get_intent_counts()

        total_examples = sum(counts.values())
        num_intents = len(counts)

        # Group by category
        by_category: dict[str, int] = {}
        for intent_code, count in counts.items():
            category = intent_code.split(".")[0]
            by_category[category] = by_category.get(category, 0) + count

        return {
            "total_examples": total_examples,
            "num_intents": num_intents,
            "by_intent": counts,
            "by_category": by_category,
        }

    async def refresh_catalog(self, filepath: str | Path) -> dict[str, int]:
        """
        Clear the catalog and reload from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Dict mapping intent codes to number of examples loaded.
        """
        await self.vector_store.clear_catalog()
        return await self.load_from_json(filepath)

    def get_core_intents(self) -> list[dict[str, str]]:
        """
        Get metadata about the 8 core MVP intents.

        Returns:
            List of intent metadata dicts.
        """
        intent_descriptions = {
            CoreIntent.ORDER_STATUS_WISMO: "Where is my order / order tracking",
            CoreIntent.ORDER_STATUS_DELIVERY_ESTIMATE: "When will my order arrive",
            CoreIntent.ORDER_MODIFY_CANCEL_ORDER: "Cancel my order",
            CoreIntent.ORDER_MODIFY_CHANGE_ADDRESS: "Change shipping address",
            CoreIntent.RETURN_EXCHANGE_RETURN_INITIATE: "Start a return",
            CoreIntent.RETURN_EXCHANGE_EXCHANGE_REQUEST: "Exchange for different item",
            CoreIntent.RETURN_EXCHANGE_REFUND_STATUS: "Check refund status",
            CoreIntent.COMPLAINT_DAMAGED_ITEM: "Item arrived damaged",
        }

        return [
            {
                "intent_code": intent.value,
                "category": intent.category,
                "intent": intent.intent_name,
                "description": description,
            }
            for intent, description in intent_descriptions.items()
        ]
