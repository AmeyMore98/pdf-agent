"""Domain models for PDF Agent business logic."""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import numpy as np


@dataclass
class Document:
    """Represents a PDF document in the system."""
    path: str
    pages: List[str]
    created_at: Optional[datetime] = None

    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return len(self.pages)

    @property
    def total_text_length(self) -> int:
        """Get the total character count across all pages."""
        return sum(len(page) for page in self.pages)


@dataclass
class DocumentChunk:
    """Represents a processed chunk of text from a document."""
    text: str
    document_path: str
    page_number: int
    chunk_index: int
    token_count: int
    embedding: Optional[np.ndarray] = None

    @property
    def has_embedding(self) -> bool:
        """Check if this chunk has an embedding."""
        return self.embedding is not None


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""
    chunk: DocumentChunk
    similarity_score: float

    @property
    def preview_text(self) -> str:
        """Get a preview of the chunk text (first 100 characters)."""
        return self.chunk.text[:100] + "..." if len(self.chunk.text) > 100 else self.chunk.text


@dataclass
class ProcessingStats:
    """Statistics from document processing."""
    document_path: str
    pages_processed: int
    chunks_created: int
    embeddings_generated: int
    processing_time_seconds: float
