"""JSON adapter using built-in json module.

Expects JSON as either:
- A list of objects (records): [{"x": 1, "y": 2}, ...]
- A dict of arrays: {"x": [1, 2, 3], "y": [4, 5, 6]}
- A list of lists: [[1, 2], [3, 4]]
- A single flat list: [1, 2, 3]
"""

import json
from typing import Any, Dict, List, Sequence

import numpy as np

from critband.io._common import filter_numerical_columns, try_parse_float


def read_json(path) -> Dict[str, Dict[str, np.ndarray]]:
    """Read JSON file into {sheet_name: {column_name: array}}."""
    with open(path, "r", encoding="utf-8") as f:
        return _parse_json(json.load(f))


def read_json_buffer(buf) -> Dict[str, Dict[str, np.ndarray]]:
    """Read JSON from a buffer."""
    text = buf.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return _parse_json(json.loads(text))


def _parse_json(data: Any) -> Dict[str, Dict[str, np.ndarray]]:
    tables: Dict[str, Dict[str, np.ndarray]] = {}

    if isinstance(data, dict):
        # Dict of arrays: {"col1": [1,2], "col2": [3,4]}
        if all(isinstance(v, list) for v in data.values()):
            columns: Dict[str, List] = {}
            for key, vals in data.items():
                columns[key] = [try_parse_float(str(v)) for v in vals]
            result = filter_numerical_columns(columns)
            if result:
                tables["Data"] = result
                return tables

        # Single flat list under a key
        for key, val in data.items():
            if isinstance(val, list) and val:
                cols = _parse_list_as_column(val)
                if cols:
                    tables[key] = cols

        if tables:
            return tables

    if isinstance(data, list):
        if not data:
            return {}

        # List of records: [{"x": 1}, {"x": 2}]
        if isinstance(data[0], dict):
            columns: Dict[str, List] = {}
            for record in data:
                for key, val in record.items():
                    if key not in columns:
                        columns[key] = []
                    columns[key].append(try_parse_float(str(val)))
            result = filter_numerical_columns(columns)
            if result:
                tables["Data"] = result
                return tables

        # Flat list of numbers
        cols = _parse_list_as_column(data)
        if cols:
            tables["Data"] = cols

    return tables


def _parse_list_as_column(data: Sequence) -> Dict[str, np.ndarray]:
    """Convert a flat list to a single-column table if numerical."""
    vals = [try_parse_float(str(v)) for v in data]
    numeric = [v for v in vals if v is not None]
    if len(vals) > 0 and len(numeric) / len(vals) >= 0.8:
        arr = np.array([v if v is not None else np.nan for v in vals])
        return {"Value": arr}
    return {}
