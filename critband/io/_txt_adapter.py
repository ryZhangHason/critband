"""TXT/TSV adapter using built-in str.split with delimiter auto-detection."""

from typing import Dict, List

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


def read_txt(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read TXT/TSV file into {sheet_name: {column_name: array}}."""
    with open(path, "r", encoding="utf-8-sig") as f:
        return _parse_txt(f.read())


def read_txt_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read TXT/TSV from a buffer."""
    text = buf.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    return _parse_txt(text)


def _detect_delimiter(header_line: str) -> str:
    """Detect the delimiter from the first line of text."""
    if "\t" in header_line:
        return "\t"
    if "|" in header_line:
        return "|"
    # Default to whitespace but prefer comma
    if "," in header_line:
        return ","
    return None  # whitespace split


def _parse_txt(text: str) -> Dict[str, Dict[str, np.ndarray]]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 1:
        return {}

    # First non-empty line is headers
    header_line = lines[0]
    delimiter = _detect_delimiter(header_line)

    if delimiter:
        headers = [h.strip() for h in header_line.split(delimiter)]
        data_lines = lines[1:]
    else:
        # Whitespace-delimited: first line might have variable columns
        headers = header_line.split()
        data_lines = lines[1:]

    columns: Dict[str, List] = {h: [] for h in headers}
    for line in data_lines:
        if delimiter:
            parts = [p.strip() for p in line.split(delimiter)]
        else:
            parts = line.split()
        for i, h in enumerate(headers):
            val = try_parse_float(parts[i] if i < len(parts) else "")
            columns[h].append(val)

    return {"Sheet1": filter_numerical_columns(columns)}
