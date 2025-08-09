"""Business services for PDF Agent domain logic."""

import time
from typing import List, Optional
from datetime import datetime

from .models import Document, DocumentChunk, SearchResult, ProcessingStats
from ..core.pdf_parser import PDFParser
from ..core.embedding_processor import EmbeddingProcessor
from ..core.vector_store import VectorStore


class DocumentProcessingService:
    """Service for processing PDF documents into searchable chunks."""

    def __init__(self, parser: Optional[PDFParser] = None,
                 embedding_processor: Optional[EmbeddingProcessor] = None):
        """Initialize the service with optional dependencies."""
        self.parser = parser or PDFParser()
        self.embedding_processor = embedding_processor or EmbeddingProcessor()

    def load_document(self, file_path: str) -> Document:
        """Load a PDF document and convert it to our domain model."""
        langchain_docs = self.parser.parse(file_path)
        pages = [doc.page_content for doc in langchain_docs]

        return Document(
            path=file_path,
            pages=pages,
            created_at=datetime.now()
        )

    def process_document(self, document: Document,
                        chunk_size: int = 1000,
                        overlap_size: int = 20) -> tuple[List[DocumentChunk], ProcessingStats]:
        """Process a document into chunks with embeddings."""
        start_time = time.time()

        # Convert domain model back to LangChain format for processing
        from langchain_core.documents import Document as LangChainDoc
        langchain_docs = [
            LangChainDoc(page_content=page, metadata={"page": i})
            for i, page in enumerate(document.pages)
        ]

        # Process through existing embedding processor
        processed_chunks = self.embedding_processor.process_pdf_pages(
            page_texts=langchain_docs,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )

        # Convert to domain models
        domain_chunks = []
        for i, chunk_data in enumerate(processed_chunks):
            domain_chunk = DocumentChunk(
                text=chunk_data.text,
                document_path=document.path,
                page_number=chunk_data.page,
                chunk_index=i,
                token_count=chunk_data.token_count,
                embedding=chunk_data.embedding
            )
            domain_chunks.append(domain_chunk)

        processing_time = time.time() - start_time

        stats = ProcessingStats(
            document_path=document.path,
            pages_processed=document.page_count,
            chunks_created=len(domain_chunks),
            embeddings_generated=len([c for c in domain_chunks if c.has_embedding]),
            processing_time_seconds=processing_time
        )

        return domain_chunks, stats


class SearchService:
    """Service for searching processed documents."""

    def __init__(self, vector_store: Optional[VectorStore] = None,
                 embedding_processor: Optional[EmbeddingProcessor] = None):
        """Initialize the service with optional dependencies."""
        self.vector_store = vector_store
        self.embedding_processor = embedding_processor or EmbeddingProcessor()

    def store_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store document chunks in the vector store."""
        if not self.vector_store:
            raise ValueError("Vector store is required for storing chunks")

        if not chunks:
            return

        # Use the first chunk's document path (they should all be the same)
        document_path = chunks[0].document_path
        self.vector_store.store_embeddings(document_path, chunks)

    def search_similar(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search for similar document chunks."""
        if not self.vector_store:
            raise ValueError("Vector store is required for searching")

        # Generate embedding for query
        query_embedding = self.embedding_processor.model.encode(query)

        # Search vector store
        raw_results = self.vector_store.search_similar(query_embedding, limit)

        # Convert to domain models
        search_results = []
        for result in raw_results:
            chunk = DocumentChunk(
                text=result['text'],
                document_path=result['document_path'],
                page_number=result['page'],
                chunk_index=0,  # Not stored in current schema
                token_count=len(result['text'].split())  # Approximate
            )

            search_result = SearchResult(
                chunk=chunk,
                similarity_score=result['similarity']
            )
            search_results.append(search_result)

        return search_results


class DocumentService:
    """High-level service that orchestrates document processing and search."""

    def __init__(self, db_url: Optional[str] = None, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the document service."""
        self.processing_service = DocumentProcessingService(
            embedding_processor=EmbeddingProcessor(model_name=model_name)
        )

        self.search_service = SearchService(
            vector_store=VectorStore(db_url) if db_url else None,
            embedding_processor=self.processing_service.embedding_processor
        )

    def process_and_store_document(self, file_path: str,
                                 chunk_size: int = 1000,
                                 overlap_size: int = 20) -> ProcessingStats:
        """Complete workflow: load, process, and store a document."""
        # Load document
        document = self.processing_service.load_document(file_path)

        # Process into chunks
        chunks, stats = self.processing_service.process_document(
            document, chunk_size, overlap_size
        )

        # Store chunks
        if chunks:
            self.search_service.store_chunks(chunks)

        return stats

    def search_documents(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search across all stored documents."""
        return self.search_service.search_similar(query, limit)

    def close(self):
        """Close any open connections."""
        if self.search_service.vector_store:
            self.search_service.vector_store.close()
