"""Simple PDF Agent - Everything in one file."""

import os
import time
import re
from typing import List, Optional
import click
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import rag_search

# Configuration
DB_URL = os.getenv("DATABASE_URL", "postgresql://root:root@localhost:5432/pdf_agent")
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
OVERLAP_SIZE = 100

# Global model - loaded once
_model = None

def get_model():
    """Get the embedding model, loading it if necessary."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
        except ImportError:
            raise Exception("sentence-transformers required: pip install sentence-transformers")
    return _model

def setup_database():
    """Set up the database table."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    with conn.cursor() as cursor:
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
            CREATE INDEX embeddings_embedding_idx
            ON embeddings USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)


    conn.close()

def normalize_text(text: str) -> str:
    """Normalize whitespace in text for better embeddings."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def process_pdf(pdf_path: str, chunk_size: int = CHUNK_SIZE, overlap_size: int = OVERLAP_SIZE):
    """Process a PDF file and store embeddings."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"File must be a PDF: {pdf_path}")

    start_time = time.time()

    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    model = get_model()
    all_texts = []
    all_pages = []

    for doc in documents:
        page_number = doc.metadata.get('page', 0) + 1

        for chunk_text in splitter.split_text(doc.page_content):
            normalized = normalize_text(chunk_text)
            all_texts.append(normalized)
            all_pages.append(page_number)

    # batch encode once
    if all_texts:
        embs = model.encode(
            all_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # good for cosine similarity
        ).astype(np.float32)

        for text, page, vec in zip(all_texts, all_pages, embs):
            chunks.append({
                'text': text,
                'embedding': vec,
                'page': page
            })


    # Store in database
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True

    with conn.cursor() as cursor:
        # Clear existing entries for this document
        cursor.execute("DELETE FROM embeddings WHERE document_path = %s", (pdf_path,))

        # Insert new embeddings
        insert_data = []
        for chunk in chunks:
            vector_str = f"[{','.join(map(str, chunk['embedding'].tolist()))}]"
            insert_data.append((pdf_path, chunk['text'], vector_str, chunk['page']))

        if insert_data:
            execute_values(
                cursor,
                "INSERT INTO embeddings (document_path, text, embedding, page) VALUES %s",
                insert_data
            )

    conn.close()

    processing_time = time.time() - start_time

    return {
        'pages_processed': len(documents),
        'chunks_created': len(chunks),
        'embeddings_generated': len(chunks),
        'processing_time_seconds': processing_time
    }

def search_documents(query: str, limit: int = 10):
    """Search for similar document chunks."""
    model = get_model()
    normalized_query = normalize_text(query)  # Add this line
    query_embedding = model.encode(normalized_query)
    vector_str = f"[{','.join(map(str, query_embedding.tolist()))}]"

    conn = psycopg2.connect(DB_URL)

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT document_path, text, page,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vector_str, vector_str, limit))

        results = cursor.fetchall()

    conn.close()

    return [
        {
            'document_path': row[0],
            'text': row[1],
            'page': row[2],
            'similarity': row[3],
            'preview': row[1][:100] + "..." if len(row[1]) > 100 else row[1]
        }
        for row in results
    ]

# CLI Interface
@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Simple PDF Agent - Process and search PDF documents."""
    pass

@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--chunk-size', default=CHUNK_SIZE, help='Characters per chunk')
@click.option('--overlap-size', default=OVERLAP_SIZE, help='Overlap between chunks')
def process(pdf_path, chunk_size, overlap_size):
    """Process a PDF file and generate embeddings."""
    try:
        click.echo(f"Setting up database...")
        setup_database()

        click.echo(f"Processing PDF: {pdf_path}")
        stats = process_pdf(pdf_path, chunk_size, overlap_size)

        click.echo("Processing completed:")
        click.echo(f"  - Pages processed: {stats['pages_processed']}")
        click.echo(f"  - Chunks created: {stats['chunks_created']}")
        click.echo(f"  - Embeddings generated: {stats['embeddings_generated']}")
        click.echo(f"  - Processing time: {stats['processing_time_seconds']:.2f}s")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('query', type=str)
@click.option('--limit', default=10, help='Maximum results')
def search(query, limit):
    """Search for similar text chunks."""
    try:
        results = search_documents(query, limit)

        if not results:
            click.echo("No results found.")
        else:
            click.echo(f"Found {len(results)} results for '{query}':")
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. [Score: {result['similarity']:.3f}] {result['preview']}")
                click.echo(f"   Source: {result['document_path']} (page {result['page']})")
                click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('query', type=str)
@click.option('--limit', default=5, help='Maximum chunks to retrieve')
@click.option('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
def answer(query, limit, model):
    """Get an AI-generated answer based on your PDF documents."""
    try:
        result = rag_search(search_documents, query, limit, model)

        click.echo(f"\nQuestion: {result['query']}")
        click.echo("=" * 50)
        click.echo(f"\nAnswer: {result['answer']}\n")
        click.echo("=" * 50)

        if result['sources']:
            click.echo("Sources:")
            for i, source in enumerate(result['sources'], 1):
                click.echo(f"{i}. {source['document_path']} (page {source['page']}) - Similarity: {source['similarity']:.3f}")
                click.echo(f"   Preview: {source['preview']}")
                click.echo()
        else:
            click.echo("No sources found.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    cli()
