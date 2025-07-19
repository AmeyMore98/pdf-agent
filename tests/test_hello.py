"""Tests for hello world functionality."""

import pytest
from pdf_agent.hello import hello_world


def test_hello_world():
    """Test hello world function."""
    result = hello_world()
    assert result == "Hello, World! Welcome to PDF Agent!"
    assert isinstance(result, str)
