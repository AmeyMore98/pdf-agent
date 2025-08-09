"""Text processing with sentence-transformers for embeddings."""

from typing import List, Dict, Any, Optional, NamedTuple
from langchain_core.documents import Document
import numpy as np
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ProcessedChunk(NamedTuple):
    """A simple data structure for processed text chunks."""
    text: str
    embedding: np.ndarray
    page: int
    token_count: int


class EmbeddingProcessor:
    """Process text using sentence-transformers for embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the processor with a sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required. Install it with: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)

    def process_pdf_pages(
        self,
        page_texts: List[Document],
        chunk_size: int = 1000,
        overlap_size: int = 20
    ) -> List[ProcessedChunk]:
        """
        Use splitter to split each page into chunks.
        Create a ProcessedChunk for each chunk with metadata from the page.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=["\n\n", "\n", " ", ""],
        )

        processed_chunks = []

        for page_doc in page_texts:
            # Split the page content into chunks
            page_chunks = splitter.split_text(page_doc.page_content)
            page_number = page_doc.metadata.get('page', 0)

            # Create ProcessedChunk for each chunk
            for chunk_text in page_chunks:
                processed_chunk = ProcessedChunk(
                    text=chunk_text,
                    embedding=self.model.encode(chunk_text),
                    page=page_number + 1,
                    token_count=len(chunk_text.split()),
                )
                processed_chunks.append(processed_chunk)

        return processed_chunks
