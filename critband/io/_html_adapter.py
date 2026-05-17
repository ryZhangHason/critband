"""HTML adapter using built-in html.parser.

Extracts <table> elements and parses them into numerical arrays.
"""

from html.parser import HTMLParser
from typing import Dict, List, Tuple

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


class _TableExtractor(HTMLParser):
    """HTML parser that extracts all tables as (label, headers, rows)."""

    def __init__(self):
        super().__init__()
        self.tables: List[Tuple[str, List[str], List[List[str]]]] = []
        self._in_table = False
        self._in_header = False
        self._in_row = False
        self._in_cell = False
        self._current_cell: List[str] = []
        self._current_row: List[str] = []
        self._current_headers: List[str] = []
        self._all_rows: List[List[str]] = []
        self._table_caption: str = ""
        self._seen_headers: bool = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._current_headers = []
            self._all_rows = []
            self._table_caption = ""
            self._seen_headers = False
        elif tag == "thead":
            self._in_header = True
        elif tag == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif tag in ("th", "td") and self._in_row:
            self._in_cell = True
            self._current_cell = []
        elif tag == "caption" and self._in_table:
            self._in_cell = True
            self._current_cell = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "table":
            self._in_table = False
            if not self._current_headers:
                self._current_headers = [
                    f"Col {i + 1}" for i in range(max((len(r) for r in self._all_rows), default=0))
                ]
            if self._current_headers and self._all_rows:
                self.tables.append(
                    (
                        self._table_caption or f"Table {len(self.tables) + 1}",
                        self._current_headers,
                        self._all_rows,
                    )
                )
            self._all_rows = []
        elif tag in ("th", "td") and self._in_cell:
            cell_text = "".join(self._current_cell).strip()
            if tag == "th":
                self._current_headers.append(cell_text)
                self._seen_headers = True
            else:
                self._current_row.append(cell_text)
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            self._in_row = False
            if self._current_row:
                self._all_rows.append(self._current_row)
            self._current_row = []
        elif tag == "thead":
            self._in_header = False
        elif tag == "caption":
            self._table_caption = "".join(self._current_cell).strip()

    def handle_data(self, data):
        if self._in_cell:
            self._current_cell.append(data)

    def handle_entityref(self, name):
        if self._in_cell:
            self._current_cell.append(f"&{name};")

    def handle_charref(self, name):
        if self._in_cell:
            self._current_cell.append(f"&#{name};")


def read_html(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read HTML file into {sheet_name: {column_name: array}}."""
    with open(path, "r", encoding="utf-8") as f:
        return _parse_html(f.read())


def read_html_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read HTML from a buffer."""
    text = buf.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return _parse_html(text)


def _parse_html(text: str) -> Dict[str, Dict[str, np.ndarray]]:
    extractor = _TableExtractor()
    extractor.feed(text)

    tables: Dict[str, Dict[str, np.ndarray]] = {}
    for label, headers, rows in extractor.tables:
        columns: Dict[str, list] = {h: [] for h in headers}
        for row in rows:
            for i, h in enumerate(headers):
                columns[h].append(try_parse_float(row[i] if i < len(row) else ""))
        result = filter_numerical_columns(columns)
        if result:
            key = label
            suffix = 2
            while key in tables:
                key = f"{label} ({suffix})"
                suffix += 1
            tables[key] = result

    return tables
