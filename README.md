# PDF Agent

A Python agent for processing and analyzing PDF documents.

## Features

- Extract text from PDF files
- Parse and analyze PDF content
- Command-line interface for easy usage
- Extensible architecture for custom processing

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
# Basic usage
pdf-agent process document.pdf

# Get help
pdf-agent --help
```

### Python API

```python
from pdf_agent import PDFProcessor

processor = PDFProcessor()
text = processor.extract_text("document.pdf")
print(text)
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
