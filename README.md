# pola — Critical Bandwidth for Bimodal Distributions

**pola** is a Python package for detecting whether a distribution is **meaningfully bimodal** using the **critical bandwidth** method in kernel density estimation (KDE).

It finds the smallest bandwidth where a KDE transitions from bimodal to unimodal — a well-established statistical test for modality.

## Quick Start

```bash
pip install pola
```

```python
import numpy as np
from pola import critical_bandwidth

# Generate bimodal data
x = np.concatenate([np.random.normal(-2, 0.5, 200),
                    np.random.normal( 2, 0.5, 200)])

h_crit, success = critical_bandwidth(x)
print(f"Critical bandwidth: {h_crit:.4f}")
print(f"Converged: {success}")
# Output: Critical bandwidth: ~0.97
#         Converged: True
```

## Features

### Core Algorithm

| Function | Description |
|----------|-------------|
| `silverman_bandwidth(x)` | Silverman's rule-of-thumb bandwidth: 1.06 · min(σ, IQR/1.34) · n^(-1/5) |
| `gaussian_kde(x, grid, h)` | Gaussian KDE evaluated on a user-specified grid |
| `critical_bandwidth(x)` | Binary search for the smallest unimodal bandwidth; returns `(h_crit, success)` |

The `critical_bandwidth` function uses binary search between lower and upper bounds (auto-computed from Silverman's rule) to find the exact transition point where the density estimate becomes unimodal.

### Multi-Format Data Loading

Read data from **9 file formats** without learning separate tools:

```python
from pola.io import read_data

# Auto-detect — just point at any supported file
x = read_data("measurements.csv")
x = read_data("survey.xlsx", sheet="Sheet1")
x = read_data("results.pdf", column="Score")
y = read_data("report.docx", column=0, return_all=False)
```

| Format | Extension | Library |
|--------|-----------|---------|
| CSV | `.csv` | Built-in `csv` (zero dependencies) |
| TSV / TXT | `.tsv`, `.txt` | Built-in (zero dependencies) |
| JSON | `.json` | Built-in `json` (zero dependencies) |
| Markdown (tables) | `.md` | Built-in parser (zero dependencies) |
| HTML (tables) | `.html`, `.htm` | Built-in `html.parser` (zero dependencies) |
| Excel | `.xlsx` | `openpyxl` |
| Excel (legacy) | `.xls` | `xlrd` |
| Word (tables) | `.docx` | `python-docx` |
| PDF (tables) | `.pdf` | `pdfplumber` |

All dependencies are **declared in pyproject.toml** — `pip install pola` installs everything automatically. No system-level tools required.

### Column / Sheet Selection

```python
from pola.io import read_data, read_buffer

# By sheet name (Excel, PDF, Markdown multi-section)
x = read_data("data.xlsx", sheet="Measurements")

# By index
x = read_data("data.xlsx", sheet=0)

# By column name
x = read_data("data.csv", column="Value")

# By column index
x = read_data("data.csv", column=1)

# Get all numerical columns as a dict
cols = read_data("data.csv", return_all=True)
# cols = {"x": array([...]), "y": array([...])}
```

### Web-Ready Buffer API

Works with file uploads from web frameworks (Flask, FastAPI, Django, Streamlit):

```python
from pola.io import read_buffer

# From a BytesIO object
buf = io.BytesIO(uploaded_file.read())
x = read_buffer(buf, filename="data.csv")
```

## Installation

```bash
# From PyPI
pip install pola

# Or using uv
uv add pola

# Development
git clone https://github.com/ryZhangHason/Polarization-CBW.git
cd Polarization-CBW
uv sync
uv run python -m pytest tests/ -v
```

### Requirements

- Python ≥ 3.10
- Runtime: numpy, scipy, openpyxl, xlrd, python-docx, pdfplumber
- Test only (not needed at runtime): pytest, pytest-cov, xlwt, reportlab

## Why pola?

- **Zero-config modality testing**: One function call tells you whether your distribution is bimodal, with a well-defined threshold.
- **Any-file input**: 9 formats from a single API. CSV, Excel, PDF, Word, JSON, HTML, Markdown — just point `read_data` at the file and go.
- **One-command install**: `pip install pola` installs everything. No system packages, no manual steps, no Tesseract OCR.
- **Pure Python dependencies**: All 4 additional libraries (openpyxl, xlrd, python-docx, pdfplumber) are pure Python — no compiled extensions.
- **Built for production and research**: Works in CLI scripts, Jupyter notebooks, and web applications equally well.

## Example: Bimodality Test

```python
import numpy as np
from pola import critical_bandwidth

# Unimodal data — the function tells you it's already unimodal
unimodal = np.random.normal(0, 1, 500)
h_crit, success = critical_bandwidth(unimodal)
print(f"Unimodal data: h_crit={h_crit:.4f}, converged={success}")
# → success may be False (already unimodal at minimum bandwidth)

# Bimodal data — critical bandwidth found
bimodal = np.concatenate([
    np.random.normal(-3, 0.5, 300),
    np.random.normal( 3, 0.5, 300)
])
h_crit, success = critical_bandwidth(bimodal)
print(f"Bimodal data: h_crit={h_crit:.4f}, converged={success}")
# → h_crit is the transition bandwidth; bandwidths below it are bimodal
```

## License

MIT
