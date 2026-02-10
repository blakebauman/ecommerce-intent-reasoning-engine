"""Tests for embedding extraction."""

import pytest

# Import directly to avoid loading spaCy via __init__
from intent_engine.extractors.embedding import EmbeddingExtractor


@pytest.fixture
def extractor() -> EmbeddingExtractor:
    """Create an embedding extractor instance."""
    return EmbeddingExtractor()


class TestEmbeddingExtractor:
    """Tests for the EmbeddingExtractor class."""

    def test_embed_returns_list(self, extractor: EmbeddingExtractor) -> None:
        """Test that embed returns a list of floats."""
        embedding = extractor.embed("Where is my order?")
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, extractor: EmbeddingExtractor) -> None:
        """Test batch embedding."""
        texts = ["Where is my order?", "Cancel my order", "Return this item"]
        embeddings = extractor.embed_batch(texts)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_similar_texts_high_similarity(self, extractor: EmbeddingExtractor) -> None:
        """Test that similar texts have high similarity."""
        emb1 = extractor.embed("Where is my order?")
        emb2 = extractor.embed("Track my order please")
        similarity = extractor.similarity(emb1, emb2)
        assert similarity > 0.5  # Should be similar

    def test_different_texts_lower_similarity(self, extractor: EmbeddingExtractor) -> None:
        """Test that different texts have lower similarity."""
        emb1 = extractor.embed("Where is my order?")
        emb2 = extractor.embed("I want to return this item")
        similarity = extractor.similarity(emb1, emb2)
        # Should be less similar than same-intent texts
        emb3 = extractor.embed("Track my package")
        sim_same = extractor.similarity(emb1, emb3)
        assert sim_same > similarity

    def test_embedding_dim_property(self, extractor: EmbeddingExtractor) -> None:
        """Test embedding_dim property."""
        assert extractor.embedding_dim == 384

    def test_similarity_self(self, extractor: EmbeddingExtractor) -> None:
        """Test that same text has similarity ~1.0."""
        emb = extractor.embed("Where is my order?")
        similarity = extractor.similarity(emb, emb)
        assert similarity > 0.99
