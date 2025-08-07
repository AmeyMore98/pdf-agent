"""Core PDF processing functionality."""

from langchain_core.documents import Document
from .pdf_parser import PDFParser
from .embedding_processor import EmbeddingProcessor, ChunkWithEmbedding
from .vector_store import VectorStore
from typing import List, Optional


class EmbeddingService:
    """Service for managing embedding creation and storage."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', db_url: Optional[str] = None):
        """Initialize the embedding service."""
        self.embedding_processor = EmbeddingProcessor(model_name=model_name)
        self.vector_store = VectorStore(db_url) if db_url else None

    def process_and_store_pdf(
        self,
        page_texts: List[Document],
        document_path: str,
        chunk_size: int = 200,
        overlap_size: int = 25
    ) -> List[ChunkWithEmbedding]:
        """Process PDF pages into embeddings and optionally store them."""
        chunks = self.embedding_processor.process_pdf_pages(
            page_texts=page_texts,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )


        if self.vector_store and chunks:
            try:
                self.vector_store.store_embeddings(document_path, chunks)
                print(f"Stored {len(chunks)} embeddings in vector store")
            except Exception as e:
                print(f"Warning: Failed to store embeddings: {e}")

        return chunks

    def search_similar(self, query_text: str, limit: int = 10) -> List[dict]:
        """Search for similar text chunks using the vector store."""
        if not self.vector_store:
            raise RuntimeError("Vector store is not available")

        query_embedding = self.embedding_processor.model.encode(query_text)
        return self.vector_store.search_similar(query_embedding, limit)

    def close(self):
        """Close the vector store connection."""
        if self.vector_store:
            self.vector_store.close()


__all__ = ["PDFParser", "EmbeddingProcessor", "ChunkWithEmbedding", "VectorStore", "EmbeddingService"]
