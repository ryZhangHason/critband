# pola — Critical Bandwidth for Bimodal Distributions

**pola** is a Python package for detecting whether a distribution is **meaningfully bimodal** using the **critical bandwidth** method in kernel density estimation (KDE).

It finds the smallest bandwidth where a KDE transitions from bimodal to unimodal — a well-established statistical test for modality.

> **Web App**: Try pola in your browser at **[qhwangantoneva.github.io/pola](https://qhwangantoneva.github.io/pola/)** — no installation required. All computation runs in-browser via Pyodide (WebAssembly). Your data never leaves your machine.

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

## Methodology

### Critical Bandwidth

The critical bandwidth is the smallest bandwidth $h_{\text{crit}}$ at which the kernel density estimate $\hat{f}(x; h)$ transitions from being **bimodal** to **unimodal**. When $h < h_{\text{crit}}$ the estimate has two modes; for $h \ge h_{\text{crit}}$ it is unimodal. A large critical bandwidth relative to the data scale indicates strong bimodality — a wide smoothing window is needed to merge the two modes.

### Silverman's Rule of Thumb

Silverman (1986) proposed a reference bandwidth for Gaussian KDE:

$$h = 1.06 \cdot \min(\sigma, \text{IQR} / 1.34) \cdot n^{-1/5}$$

This is used as the baseline for the critical bandwidth search: the solver searches for $h_{\text{crit}}$ in the interval $[h/20, 10h]$, which captures the transition for typical bimodal distributions.

### Brent-Dekker Hybrid Solver

The critical bandwidth search uses Brent's method on the continuous trough-ratio objective:

$$f(h) = \text{dip\_ratio}(h) - 0.5$$

where $\text{dip\_ratio}(h) = \text{valley\_height} / \min(\text{peak\_height})$ is a continuous measure of bimodality in $(0, 1]$. Values near 0 indicate strong bimodality; values near 1 indicate unimodality. The solver:

1. Performs a coarse binary search to bracket the root
2. Switches to Brent's method for precision convergence
3. Falls back to pure binary search if Brent fails to converge

### Related Work

- **Silverman (1981)** — The critical bandwidth test for multimodality in kernel density estimates
- **Hartigan & Hartigan (1985)** — The Dip Test of Unimodality, a complementary non-parametric approach
- **Hall & York (2001)** — On the calibration of Silverman's test for multimodality, refinements to the original method

### References

1. Silverman, B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
2. Hartigan, J.A. & Hartigan, P.M. (1985). The Dip Test of Unimodality. *The Annals of Statistics*, 13(1), 70-84.
3. Cheng, M.-Y. & Hall, P. (1998). Calibrating the excess mass and dip tests of modality. *Journal of the Royal Statistical Society: Series B*, 60(3), 579-589.

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

- Python >= 3.10
- Runtime: numpy, scipy, openpyxl, xlrd, python-docx, pdfplumber
- Test only (not needed at runtime): pytest, pytest-cov, xlwt, reportlab
- Visualization (optional): matplotlib >= 3.5

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

## Visualization Examples

Generate 3 matplotlib figures that illustrate the core algorithms:

```bash
uv sync --group viz
uv run python examples/bandwidth_demo.py
```

The demo uses well-separated bimodal data from the benchmark suite and produces:

- **`kde_bandwidth_sweep.png`** — 2×2 subplots showing KDE at 4 bandwidths (too small, Silverman, critical, too large), with peaks and troughs marked.
- **`critical_bandwidth_transition.png`** — Three overlaid KDE curves: bimodal (h_crit × 0.85), critical (h_crit), and unimodal (h_crit × 1.15), with the trough position annotated.
- **`component_decomposition.png`** — Histogram, KDE, detected Gaussian components (scaled by weight), and the trough-based separation point.

matplotlib is an optional dependency — the core package does not require it.

### Jupyter Notebook Demo

An interactive Jupyter Notebook is available for hands-on exploration:

```bash
uv sync --group viz --group jupyter
uv run jupyter notebook examples/bandwidth_demo.ipynb
```

The notebook contains 10 cells that walk through the same three visualizations with
explanatory markdown, and includes an extension exercise that compares dip ratios
across all 7 benchmark cases.

### Performance Benchmarking

Measure execution time of core functions across data sizes:

```bash
uv run python examples/benchmark_performance.py
```

Customize the number of runs and data sizes:

```bash
uv run python examples/benchmark_performance.py --runs 10 --sizes 100 500 2000
uv run python examples/benchmark_performance.py --output results.csv
```

## License

MIT
