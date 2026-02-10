"""Semantic embedding generation using sentence-transformers."""

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class EmbeddingExtractor:
    """
    Generate semantic embeddings for intent matching.

    Uses sentence-transformers with the all-MiniLM-L6-v2 model by default,
    which produces 384-dimensional embeddings optimized for semantic similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding extractor.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is all-MiniLM-L6-v2 (384 dims, fast).
        """
        self._model: SentenceTransformer | None = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings."""
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        embedding: NDArray[np.float32] = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        embeddings: NDArray[np.float32] = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Since embeddings are L2-normalized, this is just the dot product.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return float(np.dot(vec1, vec2))
