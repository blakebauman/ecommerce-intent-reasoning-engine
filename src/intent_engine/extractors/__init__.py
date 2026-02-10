"""Entity extraction and embedding modules."""

# Lazy imports to avoid loading spaCy when not needed
def __getattr__(name: str):
    if name == "EmbeddingExtractor":
        from intent_engine.extractors.embedding import EmbeddingExtractor
        return EmbeddingExtractor
    elif name == "EntityExtractor":
        from intent_engine.extractors.entity_extractor import EntityExtractor
        return EntityExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["EmbeddingExtractor", "EntityExtractor"]
