"""Command-line interface for PDF Agent."""

import click
import os
from .core.pdf_parser import PDFParser
from .core import EmbeddingService


@click.group()
@click.version_option(version="0.1.0")
def main():
    """PDF Agent - A Python agent for processing and analyzing PDF documents."""
    pass

db_url = "postgresql://root:root@localhost:5432/pdf_agent"


@main.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--chunk-size', default=200, help='Number of tokens per chunk (max depends on model, typically 256-512)')
@click.option('--overlap-size', default=25, help='Number of overlapping tokens between chunks')
def process_pdf(pdf_path, chunk_size, overlap_size):
    """Process a PDF file and generate embeddings."""
    try:
        # Parse PDF
        click.echo(f"Parsing PDF: {pdf_path}")
        parser = PDFParser()
        page_texts = parser.parse(pdf_path)
        click.echo(f"Found {len(page_texts)} pages")

        # Initialize embedding service
        service = EmbeddingService(db_url=db_url)

        # Process pages into chunks with embeddings
        document_path = os.path.abspath(pdf_path)
        chunks = service.process_and_store_pdf(
            page_texts=page_texts,
            document_path=document_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )

        # Print results
        click.echo(f"Generated {len(chunks)} chunks")
        for chunk in chunks:
            click.echo(f"Chunk: page {chunk.page}, {chunk.token_count} tokens")

        service.close()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument('query', type=str)
@click.option('--limit', default=10, help='Maximum number of results')
def search(query, limit):
    """Search for similar text chunks."""
    try:
        service = EmbeddingService(db_url=db_url)
        results = service.search_similar(query, limit)

        if not results:
            click.echo("No results found.")
        else:
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. {result['similarity']:.3f} - {result['text'][:100]}...")

        service.close()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
