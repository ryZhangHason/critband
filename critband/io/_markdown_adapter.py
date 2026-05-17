"""Markdown table adapter using built-in parsing.

Parses pipe-delimited markdown tables (| col1 | col2 |) into numerical arrays.
"""

import re
from typing import Dict, List, Tuple

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


def read_markdown(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read Markdown file into {sheet_name: {column_name: array}}."""
    with open(path, "r", encoding="utf-8") as f:
        return parse_markdown(f.read())


def read_markdown_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read Markdown from a buffer."""
    text = buf.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return parse_markdown(text)


def parse_markdown(text: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse markdown text into {table_label: {column_name: array}}."""
    sections = _extract_sections(text)
    tables: Dict[str, Dict[str, np.ndarray]] = {}

    for label, body in sections:
        parsed = _parse_single_table(body)
        if parsed:
            key = label
            suffix = 2
            while key in tables:
                key = f"{label} ({suffix})"
                suffix += 1
            tables[key] = parsed

    return tables


def _extract_sections(text: str) -> List[Tuple[str, str]]:
    """Split markdown by ## headings into (label, body) pairs."""
    lines = text.split("\n")
    sections: List[Tuple[str, str]] = []
    current_heading = None
    current_lines: List[str] = []

    for line in lines:
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            if current_heading is not None or any(line.strip() for line in current_lines):
                sections.append(
                    (
                        current_heading or "Table 1",
                        "\n".join(current_lines),
                    )
                )
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None or any(line.strip() for line in current_lines):
        sections.append(
            (
                current_heading or "Data",
                "\n".join(current_lines),
            )
        )

    if not sections:
        sections.append(("Data", text))

    return sections


def _parse_single_table(text: str) -> Dict[str, np.ndarray]:
    """Parse a single pipe-delimited markdown table."""
    lines = text.strip().split("\n")

    table_rows: List[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_rows.append(stripped)
            in_table = True
        elif in_table:
            break

    if len(table_rows) < 2:
        return {}

    headers = [h.strip() for h in table_rows[0].strip("|").split("|")]

    columns: Dict[str, list] = {h: [] for h in headers}
    for row in table_rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        for i, header in enumerate(headers):
            val = try_parse_float(cells[i] if i < len(cells) else "")
            columns[header].append(val)

    return filter_numerical_columns(columns)
