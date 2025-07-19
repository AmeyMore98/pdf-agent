"""Command-line interface for PDF Agent."""

import click
from .hello import hello_world


@click.group()
@click.version_option(version="0.1.0")
def main():
    """PDF Agent - A Python agent for processing and analyzing PDF documents."""
    pass


@main.command()
def hello():
    """Say hello from PDF Agent."""
    click.echo(hello_world())


if __name__ == "__main__":
    main()
