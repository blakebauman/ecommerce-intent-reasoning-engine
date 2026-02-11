"""Vector storage operations using pgvector."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector


@dataclass
class SimilarityMatch:
    """Result from a similarity search."""

    id: int
    intent_code: str
    category: str
    example_text: str
    similarity: float


class VectorStore:
    """
    Vector storage using PostgreSQL with pgvector extension.

    Provides similarity search for intent matching using HNSW index
    for fast approximate nearest neighbor queries.
    """

    def __init__(self, database_url: str) -> None:
        """
        Initialize the vector store.

        Args:
            database_url: PostgreSQL connection string.
        """
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Establish connection pool to the database."""
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=2,
            max_size=10,
            init=self._init_connection,
        )

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize connection with pgvector support."""
        await register_vector(conn)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def check(self) -> bool:
        """Check database connectivity (for readiness probes). Returns True if connected."""
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool."""
        if not self._pool:
            raise RuntimeError("VectorStore not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            yield conn

    async def insert_embedding(
        self,
        intent_code: str,
        category: str,
        example_text: str,
        embedding: list[float],
    ) -> int:
        """
        Insert a new intent example with its embedding.

        Args:
            intent_code: The intent code (e.g., "ORDER_STATUS.WISMO").
            category: The intent category (e.g., "ORDER_STATUS").
            example_text: The example utterance.
            embedding: The embedding vector (384 dims for MiniLM).

        Returns:
            The ID of the inserted row.
        """
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO intent_catalog (intent_code, category, example_text, embedding)
                VALUES ($1, $2, $3, $4)
                RETURNING id
                """,
                intent_code,
                category,
                example_text,
                embedding,
            )
            return int(row["id"])

    async def insert_embeddings_batch(
        self,
        records: list[tuple[str, str, str, list[float]]],
    ) -> int:
        """
        Insert multiple intent examples in a batch.

        Args:
            records: List of (intent_code, category, example_text, embedding) tuples.

        Returns:
            Number of rows inserted.
        """
        async with self.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO intent_catalog (intent_code, category, example_text, embedding)
                VALUES ($1, $2, $3, $4)
                """,
                records,
            )
            return len(records)

    async def similarity_search(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[SimilarityMatch]:
        """
        Find the most similar intent examples using cosine similarity.

        Args:
            embedding: The query embedding vector.
            top_k: Number of results to return.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of SimilarityMatch objects sorted by similarity (descending).
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id,
                    intent_code,
                    category,
                    example_text,
                    1 - (embedding <=> $1) as similarity
                FROM intent_catalog
                WHERE 1 - (embedding <=> $1) >= $3
                ORDER BY embedding <=> $1
                LIMIT $2
                """,
                embedding,
                top_k,
                min_similarity,
            )

            return [
                SimilarityMatch(
                    id=row["id"],
                    intent_code=row["intent_code"],
                    category=row["category"],
                    example_text=row["example_text"],
                    similarity=float(row["similarity"]),
                )
                for row in rows
            ]

    async def get_intent_counts(self) -> dict[str, int]:
        """Get the count of examples per intent code."""
        async with self.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT intent_code, COUNT(*) as count
                FROM intent_catalog
                GROUP BY intent_code
                ORDER BY intent_code
                """
            )
            return {row["intent_code"]: row["count"] for row in rows}

    async def delete_intent_examples(self, intent_code: str) -> int:
        """Delete all examples for a specific intent."""
        async with self.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM intent_catalog WHERE intent_code = $1",
                intent_code,
            )
            # Parse "DELETE N" result
            return int(result.split()[-1])

    async def clear_catalog(self) -> None:
        """Delete all intent examples. Use with caution."""
        async with self.acquire() as conn:
            await conn.execute("TRUNCATE intent_catalog RESTART IDENTITY")
