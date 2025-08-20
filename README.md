# Simple PDF Agent

A minimal PDF processing tool that extracts text, creates embeddings, and enables similarity search.

## Setup

```bash
pip install -r requirements-simple.txt
```

## Usage

Process a PDF:

```bash
python main.py process /path/to/document.pdf
```

Search documents:

```bash
python main.py search "your search query"
```

## Requirements

- PostgreSQL with pgvector extension

## What it does

1. **Process**: Loads PDF → splits into chunks → generates embeddings → stores in PostgreSQL
2. **Search**: Takes query → generates embedding → finds similar chunks → returns results
