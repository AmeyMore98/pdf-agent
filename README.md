# PDF Agent

A Python agent for processing and analyzing PDF documents.

## Features

- Extract text from PDF files and generate embeddings
- Store embeddings in PostgreSQL with pgvector for similarity search
- Command-line interface for processing and searching

## Installation

### Using pip

```bash
pip install -e .
```

### Development Installation

```bash
pip install -r requirements-dev.txt
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Process PDF and store embeddings
pdf-agent process-pdf document.pdf --db-url postgresql://user:pass@host/db

# Search for similar content
pdf-agent search "machine learning" --db-url postgresql://user:pass@host/db

# Get help
pdf-agent --help
```

### Python API

```python
from pdf_agent.core import PDFParser, EmbeddingProcessor

# Parse PDF and process embeddings
parser = PDFParser()
page_texts = parser.parse("document.pdf")

processor = EmbeddingProcessor(db_url="postgresql://user:pass@host/db")
chunks = processor.process_pdf_pages(page_texts, document_path="document.pdf")

# Search for similar content
results = processor.search_similar("machine learning")
```

## Setup

Install PostgreSQL with pgvector extension:

```bash
# Run PostgreSQL with pgvector
docker run --name pdf-agent-postgres \
  -e POSTGRES_DB=pdf_agent \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d pgvector/pgvector:pg16
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd pdf-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
```

### Type Checking

```bash
mypy src/
```

## Project Structure

```
pdf-agent/
├── src/
│   └── pdf_agent/
│       ├── __init__.py
│       ├── core/
│       ├── cli.py
│       └── utils/
├── tests/
├── docs/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```
