"""Command-line interface for PDF Agent."""

import click
import os
from .domain import DocumentService


@click.group()
@click.version_option(version="0.1.0")
def main():
    """PDF Agent - A Python agent for processing and analyzing PDF documents."""
    pass

db_url = "postgresql://root:root@localhost:5432/pdf_agent"


@main.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--chunk-size', default=1000, help='Number of characters per chunk')
@click.option('--overlap-size', default=20, help='Number of overlapping characters between chunks')
def process_pdf(pdf_path, chunk_size, overlap_size):
    """Process a PDF file and generate embeddings."""
    try:
        click.echo(f"Processing PDF: {pdf_path}")

        # Initialize document service
        service = DocumentService(db_url=db_url)

        # Process and store document
        document_path = os.path.abspath(pdf_path)
        stats = service.process_and_store_document(
            file_path=document_path,
            chunk_size=chunk_size,
            overlap_size=overlap_size
        )

        # Print results
        click.echo(f"Processing completed:")
        click.echo(f"  - Pages processed: {stats.pages_processed}")
        click.echo(f"  - Chunks created: {stats.chunks_created}")
        click.echo(f"  - Embeddings generated: {stats.embeddings_generated}")
        click.echo(f"  - Processing time: {stats.processing_time_seconds:.2f}s")

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
        service = DocumentService(db_url=db_url)
        results = service.search_documents(query, limit)

        if not results:
            click.echo("No results found.")
        else:
            click.echo(f"Found {len(results)} results for '{query}':")
            for i, result in enumerate(results, 1):
                click.echo(f"{i}. [Score: {result.similarity_score:.3f}] {result.preview_text}")
                click.echo(f"   Source: {result.chunk.document_path} (page {result.chunk.page_number})")
                click.echo()

        service.close()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()
