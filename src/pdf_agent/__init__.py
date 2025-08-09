"""PDF Agent - A Python agent for processing and analyzing PDF documents."""

from .domain import DocumentService, Document, DocumentChunk, SearchResult

__version__ = "0.1.0"
__author__ = "PDF Agent Team"
__email__ = "amey8976@gmail.com"

# Primary API exports
__all__ = ["DocumentService", "Document", "DocumentChunk", "SearchResult"]
