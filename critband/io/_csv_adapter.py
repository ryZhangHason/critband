"""CSV adapter using built-in csv module."""

import csv
import io
from typing import Dict, List

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


def read_csv(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read CSV file into {sheet_name: {column_name: array}}."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return _parse_csv(f)


def read_csv_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read CSV from a buffer."""
    text = buf.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    return _parse_csv(io.StringIO(text))


def _parse_csv(text_io) -> Dict[str, Dict[str, np.ndarray]]:
    reader = csv.DictReader(text_io)
    rows = list(reader)
    if not rows or not reader.fieldnames:
        return {}

    columns: Dict[str, List] = {h: [] for h in reader.fieldnames}
    for row in rows:
        for h in reader.fieldnames:
            columns[h].append(try_parse_float(row.get(h, "")))

    return {"Sheet1": filter_numerical_columns(columns)}
