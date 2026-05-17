"""
Dispatcher for critband.io adapters.

Routes file paths to the appropriate format-specific adapter based on file
extension, then applies shared column-selection logic.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


class DataReadError(ValueError):
    """
    Exception raised when input data cannot be read or parsed.

    Reasons include:
    - Unsupported file format
    - No numerical data found in any sheet/table
    - Empty or corrupted file
    - Out-of-range sheet or column index
    - Nonexistent sheet or column name
    """

    pass


def read_data(
    path: Union[str, Path],
    sheet: Optional[Union[str, int]] = None,
    column: Optional[Union[str, int]] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Read numerical data from a file into numpy arrays.

    Parameters
    ----------
    path : str or Path
        Path to the input file.
    sheet : str or int, optional
        For multi-sheet formats (XLSX, XLS), select a specific sheet by name
        or index (0-based). Defaults to the first sheet with numerical data.
    column : str or int, optional
        Column name or 0-based index. If None (default), returns the first
        fully numerical column found in the selected sheet.
    return_all : bool, optional
        If True, returns a dict[str, np.ndarray] of all numerical columns
        in the selected sheet/buffer, keyed by column name.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    adapter = _EXTENSION_MAP.get(ext)
    if adapter is None:
        raise DataReadError(
            f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(_EXTENSION_MAP))}"
        )

    tables = adapter(path)
    return _select_columns(tables, sheet, column, return_all)


def read_buffer(
    buf,
    filename: str,
    sheet: Optional[Union[str, int]] = None,
    column: Optional[Union[str, int]] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Read numerical data from a file-like object (e.g. BytesIO from web upload).

    Parameters
    ----------
    buf : file-like object (readable in binary mode)
        The buffer to read from.
    filename : str
        The original filename (used for format detection via extension).
    sheet, column, return_all : same as read_data.

    Returns
    -------
    np.ndarray or dict[str, np.ndarray]
    """
    ext = Path(filename).suffix.lower()
    adapter = _BUFFER_ADAPTERS.get(ext)
    if adapter is None:
        raise DataReadError(
            f"Unsupported file format '{ext}' for buffer read. "
            f"Supported: {', '.join(sorted(_BUFFER_ADAPTERS))}"
        )

    tables = adapter(buf)
    return _select_columns(tables, sheet, column, return_all)


def _select_columns(
    tables: Dict[str, Dict[str, np.ndarray]],
    sheet: Optional[Union[str, int]] = None,
    column: Optional[Union[str, int]] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Shared column selection logic used by both read_data and read_buffer."""
    if not tables:
        raise DataReadError("No tables found. The file may contain no tabular data.")

    table_keys = list(tables.keys())

    if sheet is not None:
        if isinstance(sheet, int):
            if sheet < 0 or sheet >= len(table_keys):
                raise DataReadError(
                    f"Sheet index {sheet} is out of range. "
                    f"Available sheets ({len(table_keys)}): {table_keys}"
                )
            selected_key = table_keys[sheet]
        else:
            if sheet not in tables:
                raise DataReadError(f"Sheet '{sheet}' not found. Available sheets: {table_keys}")
            selected_key = sheet
    else:
        selected_key = _first_sheet_with_numerical_data(tables)
        if selected_key is None:
            raise DataReadError("No numerical data found in any sheet/table.")

    columns = tables[selected_key]

    if return_all:
        return columns

    if column is not None:
        if isinstance(column, int):
            col_keys = list(columns.keys())
            if column < 0 or column >= len(col_keys):
                raise DataReadError(
                    f"Column index {column} is out of range. "
                    f"Available columns ({len(col_keys)}): {col_keys}"
                )
            col_name = col_keys[column]
        else:
            if column not in columns:
                raise DataReadError(
                    f"Column '{column}' not found. Available columns: {list(columns.keys())}"
                )
            col_name = column
    else:
        col_name = _first_numerical_column(columns)
        if col_name is None:
            raise DataReadError(f"No numerical columns found in sheet '{selected_key}'.")

    return columns[col_name]


def _first_sheet_with_numerical_data(tables: Dict[str, Dict[str, np.ndarray]]) -> Optional[str]:
    """Return the key of the first table containing at least one numerical column."""
    for key, columns in tables.items():
        if columns:
            return key
    return None


def _first_numerical_column(columns: Dict[str, np.ndarray]) -> Optional[str]:
    """Return the name of the first numerical column in a table."""
    for name in columns:
        return name
    return None


def _parse_markdown_tables(text: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Parse markdown pipe tables into {label: {col: array}}."""
    from critband.io._markdown_adapter import parse_markdown

    return parse_markdown(text)


# Lazy imports for adapters (loaded only when needed)
_EXTENSION_MAP: Dict[str, callable] = {}
_BUFFER_ADAPTERS: Dict[str, callable] = {}


def _lazy_load_ext(ext: str, module: str, func: str, buf_func: Optional[str] = None):
    """Register a path-based adapter and optionally a buffer-based adapter."""

    def adapter(path):
        mod = __import__(f"critband.io._{module}", fromlist=[func])
        return getattr(mod, func)(path)

    _EXTENSION_MAP[ext] = adapter
    if buf_func:

        def buf_adapter(buf):
            mod = __import__(f"critband.io._{module}", fromlist=[buf_func])
            return getattr(mod, buf_func)(buf)

        _BUFFER_ADAPTERS[ext] = buf_adapter


def _init_adapters():
    """Initialize all adapter registrations (called once at module import)."""
    # --- zero-dep adapters ---
    _lazy_load_ext(".csv", "csv_adapter", "read_csv", "read_csv_buffer")
    _lazy_load_ext(".tsv", "txt_adapter", "read_txt", "read_txt_buffer")
    _lazy_load_ext(".txt", "txt_adapter", "read_txt", "read_txt_buffer")
    _lazy_load_ext(".json", "json_adapter", "read_json", "read_json_buffer")

    # --- markdown tables (built-in parser) ---
    _lazy_load_ext(".md", "markdown_adapter", "read_markdown", "read_markdown_buffer")

    # --- library-based adapters ---
    _lazy_load_ext(".xlsx", "xlsx_adapter", "read_xlsx", "read_xlsx_buffer")
    _lazy_load_ext(".xls", "xls_adapter", "read_xls", "read_xls_buffer")
    _lazy_load_ext(".docx", "docx_adapter", "read_docx", "read_docx_buffer")
    _lazy_load_ext(".html", "html_adapter", "read_html", "read_html_buffer")
    _lazy_load_ext(".htm", "html_adapter", "read_html", "read_html_buffer")
    _lazy_load_ext(".pdf", "pdf_adapter", "read_pdf", "read_pdf_buffer")


_init_adapters()
