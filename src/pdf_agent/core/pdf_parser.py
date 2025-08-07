"""PDF parsing functionality for extracting text from PDF documents."""

import os
from typing import List
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document

class PDFParser:
    """A class for parsing PDF files and extracting text content."""

    def __init__(self):
        """Initialize the PDFParser."""
        pass

    def parse(self, file_path: str) -> List[Document]:
        """
        Parse a PDF file and extract raw text from each page.

        Args:
            file_path (str): Path to the PDF file to parse.

        Returns:
            List[str]: A list containing the raw text content of each page.
                      Each element in the list corresponds to one page.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ValueError: If the file is not a valid PDF.
            Exception: For other PDF processing errors.
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Validate file extension
        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {file_path}")

        try:
            loader = PyMuPDFLoader(file_path)
            return loader.load_and_split()
        except Exception as e:
            raise Exception(f"Error processing PDF file '{file_path}': {str(e)}")
