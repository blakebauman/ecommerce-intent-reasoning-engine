#!/usr/bin/env python
"""Seed the intent catalog with examples."""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intent_engine.config import get_settings
from intent_engine.extractors.embedding import EmbeddingExtractor
from intent_engine.storage.intent_catalog import IntentCatalogStore
from intent_engine.storage.vector_store import VectorStore


async def main() -> None:
    """Seed the intent catalog."""
    parser = argparse.ArgumentParser(description="Seed intent catalog with examples")
    parser.add_argument(
        "--examples",
        type=str,
        default="data/intent_examples.json",
        help="Path to intent examples JSON file",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear existing catalog before seeding",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    examples_path = project_root / args.examples

    if not examples_path.exists():
        print(f"Examples file not found: {examples_path}")
        sys.exit(1)

    # Get settings
    settings = get_settings()

    print(f"Connecting to database: {settings.database_url[:50]}...")

    # Initialize components
    vector_store = VectorStore(settings.database_url)
    await vector_store.connect()

    embedding_extractor = EmbeddingExtractor(model_name=settings.embedding_model)
    print(f"Using embedding model: {settings.embedding_model}")
    print(f"Embedding dimension: {embedding_extractor.embedding_dim}")

    catalog_store = IntentCatalogStore(
        vector_store=vector_store,
        embedding_extractor=embedding_extractor,
    )

    # Get current stats
    try:
        current_stats = await catalog_store.get_catalog_stats()
        print(f"\nCurrent catalog: {current_stats['total_examples']} examples")
    except Exception:
        current_stats = {"total_examples": 0}
        print("\nCatalog is empty or not initialized")

    # Refresh if requested
    if args.refresh:
        print("\nClearing existing catalog...")
        await vector_store.clear_catalog()

    # Load examples
    print(f"\nLoading examples from: {examples_path}")

    if args.refresh:
        counts = await catalog_store.refresh_catalog(examples_path)
    else:
        counts = await catalog_store.load_from_json(examples_path)

    print(f"\nLoaded {sum(counts.values())} examples across {len(counts)} intents:")
    for intent_code, count in sorted(counts.items()):
        print(f"  {intent_code}: {count}")

    # Show final stats
    final_stats = await catalog_store.get_catalog_stats()
    print(f"\nFinal catalog: {final_stats['total_examples']} total examples")

    # Close connection
    await vector_store.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
