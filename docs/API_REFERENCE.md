# pola API Reference

## Module: `pola` — Core Algorithm

### `silverman_bandwidth(x)`

Calculate Silverman's rule-of-thumb bandwidth for univariate data.

```python
def silverman_bandwidth(x: np.ndarray) -> float
```

**Parameters:** `x` — 1D array of input data points.

**Returns:** float — Silverman's rule-of-thumb bandwidth: `1.06 * min(std, IQR/1.34) * n^(-1/5)`. For a single data point returns 1.0. For constant data returns 0.01.

**Example:**
```python
import numpy as np
from pola import silverman_bandwidth

x = np.random.normal(0, 1, 1000)
h = silverman_bandwidth(x)  # ≈ 0.26
```

---

### `gaussian_kde(x, grid, h)`

Gaussian kernel density estimation using vectorized broadcasting.

```python
def gaussian_kde(x: np.ndarray, grid: np.ndarray, h: float) -> np.ndarray
```

**Parameters:** `x` — 1D input data. `grid` — points where density is evaluated. `h` — bandwidth.

**Returns:** np.ndarray — density estimates at each grid point.

**Example:**
```python
grid = np.linspace(-3, 3, 100)
density = gaussian_kde(x, grid, h=0.5)
```

---

### `count_modes(x, h, grid_points, prominence)`

Count the number of modes in a kernel density estimate at a given bandwidth.

```python
def count_modes(
    x: np.ndarray,
    h: float,
    grid_points: int | None = None,
    prominence: float = 0.01,
) -> int
```

**Parameters:**
- `x` — 1D input data.
- `h` — bandwidth.
- `grid_points` — evaluation grid resolution (default: adaptive, `min(800, max(200, n//10))`).
- `prominence` — minimum peak prominence as a fraction of max KDE height (default 0.01).

**Returns:** int — number of detected modes.

---

### `critical_bandwidth(x, h_min, h_max, tol, max_iter, method)`

Calculate the critical bandwidth — the smallest bandwidth where the KDE becomes unimodal.

```python
def critical_bandwidth(
    x: np.ndarray,
    h_min: float | None = None,
    h_max: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    method: str = "auto",
) -> tuple[float, bool]
```

**Parameters:**
- `x` — 1D input data.
- `h_min` — lower search bound (default: Silverman / 20).
- `h_max` — upper search bound (default: Silverman * 10).
- `tol` — convergence tolerance.
- `max_iter` — maximum iterations.
- `method` — search method: `"auto"` (default, hybrid Brent + binary), `"binary"` (pure binary search), `"brent"` (pure Brent).

**Returns:** `(h_crit, success)` — critical bandwidth value and convergence flag.

**Example:**
```python
from pola import critical_bandwidth
import numpy as np

x = np.concatenate([np.random.normal(-2, 0.5, 200),
                    np.random.normal( 2, 0.5, 200)])
h_crit, ok = critical_bandwidth(x)
```

**Notes:**
- `method="auto"` performs a coarse binary search to bracket the root, then switches to Brent's method (`scipy.optimize.brentq`) on the continuous trough-ratio objective `f(h) = dip_ratio(h) - 0.5`. Falls back to pure binary search if Brent fails.
- If already unimodal at `h_min`, returns `(h_min, False)`.
- If still bimodal at `h_max`, returns `(h_max, False)`.

---

### `find_trough(x, h, grid_points, prominence, refine)`

Find the trough (lowest point) between the two most prominent KDE modes.

```python
def find_trough(
    x: np.ndarray,
    h: float,
    grid_points: int | None = None,
    prominence: float = 0.01,
    refine: bool = True,
) -> float | None
```

**Parameters:**
- `x`, `h`, `grid_points`, `prominence` — same as `count_modes`.
- `refine` — if True, use quadratic interpolation for sub-grid precision.

**Returns:** x-coordinate of the trough, or None if fewer than 2 peaks detected.

---

### `detect_components(x, h_factor, grid_points)`

Decompose a bimodal distribution into two component Gaussians.

```python
def detect_components(
    x: np.ndarray,
    h_factor: float = 0.85,
    grid_points: int | None = None,
) -> BimodalDecomposition
```

**Parameters:**
- `x` — 1D input data.
- `h_factor` — multiplier on critical bandwidth for KDE evaluation (default 0.85). Lower values produce clearer troughs.
- `grid_points` — KDE grid resolution.

**Returns:** `BimodalDecomposition` — see dataclass below.

**Raises:** `ValueError` if fewer than 2 KDE peaks are detected.

**Algorithm:**
1. Computes `h_crit` via `critical_bandwidth`.
2. Evaluates KDE at `h_analysis = h_crit * h_factor`.
3. Finds the two highest peaks and the trough between them.
4. Splits data at the trough and computes sample mean, std, weight per side.

---

### `Component` (dataclass)

Estimated parameters of a single Gaussian component in a bimodal mixture.

```python
@dataclass
class Component:
    mean: float       # Sample mean of the component data
    std: float        # Sample standard deviation (ddof=1)
    weight: float     # Proportion of total samples in this component
```

---

### `BimodalDecomposition` (dataclass)

Complete decomposition of a bimodal distribution into two components.

```python
@dataclass
class BimodalDecomposition:
    critical_bandwidth: float   # h_crit from critical_bandwidth()
    component1: Component       # Component with lower mean
    component2: Component       # Component with higher mean
    separation_point: float     # Trough x-coordinate between modes
    dip_ratio: float            # valley_height / min(peak_height) ∈ (0, 1]
```

**dip_ratio interpretation:** near 0 = strongly bimodal (deep trough), near 1 = barely bimodal (shallow trough).

---

## Module: `pola.io` — Data Loading

### `read_data(path, sheet, column, return_all)`

Read numerical data from a file into numpy arrays. Format is auto-detected from the file extension.

```python
def read_data(
    path: str | Path,
    sheet: str | int | None = None,
    column: str | int | None = None,
    return_all: bool = False,
) -> np.ndarray | dict[str, np.ndarray]
```

**Parameters:**
- `path` — file path. Supported extensions: `.csv`, `.tsv`, `.txt`, `.json`, `.md`, `.xlsx`, `.xls`, `.docx`, `.html`, `.htm`, `.pdf`.
- `sheet` — for multi-sheet formats (XLSX, XLS, PDF, Markdown), select by name or 0-based index.
- `column` — column name or 0-based index.
- `return_all` — if True, returns `dict[str, np.ndarray]` of all numerical columns.

**Returns:** 1D numpy array (default) or dict of column name → array (if `return_all=True`).

**Raises:** `FileNotFoundError`, `DataReadError` (no numerical data, unsupported format, etc.).

**Example:**
```python
from pola.io import read_data

# Auto-detect from extension
x = read_data("measurements.csv")
y = read_data("survey.xlsx", sheet="Sheet1", column="Score")
cols = read_data("data.json", return_all=True)
```

---

### `read_buffer(buf, filename, sheet, column, return_all)`

Read numerical data from a file-like object (e.g. `BytesIO` from web upload).

```python
def read_buffer(
    buf: IO[bytes],
    filename: str,
    sheet: str | int | None = None,
    column: str | int | None = None,
    return_all: bool = False,
) -> np.ndarray | dict[str, np.ndarray]
```

**Parameters:**
- `buf` — file-like object readable in binary mode.
- `filename` — original filename (used for format detection via extension).
- `sheet`, `column`, `return_all` — same as `read_data`.

**Returns:** same as `read_data`.

**Example (Flask):**
```python
from pola.io import read_buffer
import io

uploaded = request.files["data"]
buf = io.BytesIO(uploaded.read())
x = read_buffer(buf, filename=uploaded.filename)
```

---

### `DataReadError`

Exception raised when input data cannot be read or contains no usable numerical data.

```python
class DataReadError(ValueError)
```

Inherits from `ValueError`. Raised by `read_data` and `read_buffer` for:
- Unsupported file formats.
- Empty or unparseable files.
- No numerical columns found.
- Out-of-range sheet or column indices.
- Nonexistent sheet or column names.

---

## Supported File Formats

| Format | Extension | Library | Buffer Support |
|--------|-----------|---------|----------------|
| CSV | `.csv` | Built-in `csv` | Yes |
| TSV / TXT | `.tsv`, `.txt` | Built-in | Yes |
| JSON | `.json` | Built-in `json` | Yes |
| Markdown (tables) | `.md` | Built-in parser | Yes |
| HTML (tables) | `.html`, `.htm` | Built-in `html.parser` | Yes |
| Excel | `.xlsx` | `openpyxl` | Yes |
| Excel (legacy) | `.xls` | `xlrd` | Yes |
| Word (tables) | `.docx` | `python-docx` | Yes |
| PDF (tables) | `.pdf` | `pdfplumber` | Yes |

All dependencies are declared in `pyproject.toml` and installed automatically with `pip install pola`.
