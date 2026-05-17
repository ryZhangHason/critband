"""PDF adapter using pdfplumber.

Extracts tables from PDF pages. Works best with PDFs containing
clearly defined table structures.
"""

from typing import Dict, List

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


def read_pdf(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read PDF file into {page_label: {column_name: array}}."""
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        return _parse_pdf(pdf)


def read_pdf_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read PDF from a buffer."""
    import pdfplumber

    with pdfplumber.open(buf) as pdf:
        return _parse_pdf(pdf)


def _parse_pdf(pdf) -> Dict[str, Dict[str, np.ndarray]]:
    tables: Dict[str, Dict[str, np.ndarray]] = {}
    table_counter = 0

    for page_num, page in enumerate(pdf.pages):
        page_tables = page.extract_tables()
        for pdf_table in page_tables:
            if not pdf_table or len(pdf_table) < 2:
                continue

            # First row is headers
            headers = [str(c).strip() if c else f"Col{j + 1}" for j, c in enumerate(pdf_table[0])]
            if not headers:
                continue

            columns: Dict[str, List] = {h: [] for h in headers}
            for row in pdf_table[1:]:
                for j, h in enumerate(headers):
                    cell_str = str(row[j]).strip() if j < len(row) and row[j] else ""
                    columns[h].append(try_parse_float(cell_str))

            result = filter_numerical_columns(columns)
            if result:
                table_counter += 1
                label = f"Page {page_num + 1} - Table {table_counter}"
                tables[label] = result

    return tables
