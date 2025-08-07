"""Vector store implementation using PostgreSQL with pgvector extension."""

import os
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values
import numpy as np


class VectorStore:
    """PostgreSQL vector store using pgvector extension."""

    def __init__(self, db_url: Optional[str] = None):
        """Initialize the vector store with database URL."""
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "postgresql://postgres@localhost/pdf_agent")

        self.db_url = db_url
        self.connection = None
        self._connect()
        self._create_table()

    def _connect(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(self.db_url)
            self.connection.autocommit = True
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL database: {e}")

    def _create_table(self):
        """Create the embeddings table if it doesn't exist."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        document_path TEXT NOT NULL,
                        text TEXT NOT NULL,
                        embedding VECTOR(384),
                        page INTEGER NOT NULL
                    );
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
                    ON embeddings USING ivfflat (embedding vector_cosine_ops);
                """)
        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to create table: {e}")

    def store_embeddings(self, document_path: str, chunks_with_embeddings: List[Any]):
        """Store embeddings for a document in the database."""
        if not chunks_with_embeddings:
            return

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM embeddings WHERE document_path = %s", (document_path,))

                insert_data = []
                for chunk in chunks_with_embeddings:
                    # Convert numpy array to pgvector format string
                    vector_str = f"[{','.join(map(str, chunk.embedding.tolist()))}]"
                    insert_data.append((
                        document_path,
                        chunk.text,
                        vector_str,
                        chunk.page,
                    ))

                execute_values(
                    cursor,
                    "INSERT INTO embeddings (document_path, text, embedding, page) VALUES %s",
                    insert_data
                )
        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to store embeddings: {e}")

    def search_similar(self, query_embedding: np.ndarray, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity."""
        try:
            with self.connection.cursor() as cursor:
                # Convert numpy array to pgvector format string
                vector_str = f"[{','.join(map(str, query_embedding.tolist()))}]"

                cursor.execute("""
                    SELECT document_path, text, page,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (vector_str, vector_str, limit))

                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in results]
        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to search embeddings: {e}")

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
