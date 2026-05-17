"""
pola.io: Input adapters for reading numerical data from various file formats.

Supported formats: CSV, TXT/TSV, JSON, XLSX, XLS, Markdown, DOCX, HTML, PDF.
All dependencies are declared in pyproject.toml and installed automatically.
"""

from pola.io._base import DataReadError, read_buffer, read_data

__all__ = ["read_data", "read_buffer", "DataReadError"]
