"""
Shared utilities for critband.io adapters.

Provides functions for parsing numerical values from cell text and filtering
columns that contain sufficient numerical data.
"""

from typing import Dict, List, Optional

import numpy as np


def try_parse_float(cell: str) -> Optional[float]:
    """Attempt to parse a cell value as float. Returns None on failure."""
    cell = cell.strip()
    if not cell:
        return None
    try:
        return float(cell)
    except (ValueError, TypeError):
        return None


def is_numerical_column(values: List[Optional[float]], threshold: float = 0.8) -> bool:
    """Check if a column has at least threshold fraction of numerical values."""
    if len(values) == 0:
        return False
    numeric_count = sum(1 for v in values if v is not None)
    return numeric_count / len(values) >= threshold


def filter_numerical_columns(
    columns: Dict[str, List[Optional[float]]], threshold: float = 0.8
) -> Dict[str, np.ndarray]:
    """Filter columns to only those with sufficient numerical data, converting to arrays."""
    result: Dict[str, np.ndarray] = {}
    for header, vals in columns.items():
        if is_numerical_column(vals, threshold):
            arr = np.array([v if v is not None else np.nan for v in vals])
            result[header] = arr
    return result
