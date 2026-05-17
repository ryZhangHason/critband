"""XLS adapter using xlrd."""

from typing import Dict, List

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


def read_xls(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read XLS file into {sheet_name: {column_name: array}}."""
    import xlrd

    wb = xlrd.open_workbook(path)
    return _parse_xls_workbook(wb)


def read_xls_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read XLS from a buffer."""
    import xlrd

    wb = xlrd.open_workbook(file_contents=buf.read())
    return _parse_xls_workbook(wb)


def _cell_value(cell) -> float:
    """Extract float value from an xlrd cell."""
    import xlrd

    if cell.ctype == xlrd.XL_CELL_EMPTY:
        return None
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        return cell.value
    if cell.ctype == xlrd.XL_CELL_TEXT:
        return try_parse_float(cell.value)
    # Try string representation as fallback
    return try_parse_float(str(cell.value))


def _parse_xls_workbook(wb) -> Dict[str, Dict[str, np.ndarray]]:
    import xlrd

    tables: Dict[str, Dict[str, np.ndarray]] = {}
    for sheet_idx in range(wb.nsheets):
        ws = wb.sheet_by_index(sheet_idx)
        if ws.nrows < 1:
            continue

        # First row as headers
        headers = [
            str(ws.cell_value(0, c)) if ws.cell_type(0, c) != xlrd.XL_CELL_EMPTY else f"Col{c + 1}"
            for c in range(ws.ncols)
        ]
        if not headers:
            continue

        columns: Dict[str, List] = {h: [] for h in headers}
        for r in range(1, ws.nrows):
            for c, h in enumerate(headers):
                columns[h].append(_cell_value(ws.cell(r, c)))

        result = filter_numerical_columns(columns)
        if result:
            tables[ws.name] = result

    return tables
