---
title: "critband: Critical Bandwidth Analysis for Bimodal Distributions in Python"
tags:
  - Python
  - statistics
  - kernel density estimation
  - bimodality
  - critical bandwidth
authors:
  - name: Ruiyu Zhang
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Qihao Wang
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
  - name: The University of Hong Kong
    index: 1
  - name: Independent Researcher
    index: 2
date: 15 May 2026
bibliography: paper.bib
---

# Summary

**critband** is a Python package for detecting and analyzing multimodal distributions using the critical bandwidth method in kernel density estimation (KDE). The critical bandwidth $h_{\text{crit}}$ is the smallest bandwidth at which the KDE transitions from $k$ modes to $k-1$ modes — a well-established statistical test introduced by Silverman (1981). The package provides a flexible interface: `critical_bandwidth(x, k=2, return_ci=False) → (h_crit, success)` (or `(h_crit, success, ci)` when `return_ci=True`), returning the exact transition point for the specified number of modes $k$ (default $k=2$, supporting arbitrary $k \ge 2$), and a convergence flag. critband is the first Python package dedicated to this method. Beyond the core solver, the package offers a comprehensive suite of modality analysis tools: mode detection with arbitrary-$k$ support (`find_modes`), bimodality strength quantification (`bimodality_strength`), excess mass testing (`excess_mass`), Silverman's bootstrap test (`silverman_test`), bootstrap confidence intervals for $h_{\text{crit}}$ (via `return_ci=True`), bimodal decomposition into Gaussian components (`detect_components`), and a multi-format I/O subsystem (`critband.io`) that reads numerical data from nine file formats — CSV, TSV, JSON, Markdown, HTML, XLSX, XLS, DOCX, and PDF — through a unified API. Hartigan's dip test is retained as a complementary unimodality check, not as a headline differentiator. The package is designed for researchers and practitioners who need a principled, reproducible test for bimodality without leaving the Python ecosystem.

# Statement of Need

Detecting whether a univariate distribution is meaningfully bimodal is a recurring question across ecology, economics, genomics, and astronomy. Silverman's critical bandwidth test (1981) provides a principled answer, but no dedicated Python implementation has existed until now. Existing tools are scattered across the R ecosystem — the `multimode` package implements Silverman's test and the excess mass test, `diptest` implements Hartigan's dip test, and `ks` provides kernel smoothing with modality testing. None of these are accessible to Python users without running a separate R environment or using subprocess bridges.

The dip test (Hartigan & Hartigan, 1985) is a complementary nonparametric approach but measures departure from unimodality in the cumulative distribution function rather than the density, and is less sensitive to certain bimodal structures. critband fills this gap with a pure-Python implementation of the critical bandwidth method that includes: (1) binary search and Brent's method solvers with automatic selection, (2) arbitrary-$k$ mode detection for multimodal distributions, (3) bootstrap percentile confidence intervals and Silverman's bootstrap $p$-value for uncertainty quantification, (4) bimodal component decomposition with trough detection, (5) excess mass and bimodality strength metrics for interpretable diagnostics, and (6) transparent multi-format data loading that eliminates a common friction point in applied analysis. Hartigan's dip test is retained as a complementary check rather than a headline feature.

The package has been validated against equivalent R implementations (`multimode`, `diptest`, and `ks`) across all 12 benchmark cases. critband's `critical_bandwidth()` produces $h_{\text{crit}}$ values that agree with R's `modetest` within machine precision for strongly separated components, and correctly flags near-unimodal cases with small $h_{\text{crit}}$ values where R's dip test and Silverman's test return non-significant $p$-values. Performance benchmarks show critband is 3--10$\times$ faster than R's `modetest` for critical bandwidth computation and 4--100$\times$ faster for mode counting, while the dip test runs at comparable speeds in both environments.

The package requires only pure Python dependencies — `numpy`, `scipy`, `openpyxl`, `python-docx`, `xlrd`, and `pdfplumber` — and installs with a single `pip install critband` command.

# Implementation and Validation

The core algorithm finds the critical bandwidth via search on KDE mode counts. Given input data $x$, the solver first computes Silverman's rule-of-thumb bandwidth $h_s = 1.06 \cdot \min(\hat{\sigma}, \text{IQR}/1.34) \cdot n^{-1/5}$ and sets the search interval to $[h_s / 20, 10 \cdot h_s]$, which empirically brackets the transition for typical multimodal distributions. Three solver methods are available: `"binary"` (pure binary search), `"brent"` (Brent's method on a continuous trough-ratio objective), and `"auto"` (default; coarse binary search followed by Brent, falling back to binary for weak/edge cases). Mode counting uses `scipy.signal.find_peaks` with a prominence threshold to filter spurious modes from numerical noise. At each iteration, the KDE is evaluated on an adaptive grid whose resolution scales with data size: $\max(800, \min(5000, n/2))$ points. For large samples ($n > 5000$), the KDE computation automatically switches to an FFT-accelerated path for $\mathcal{O}(g \log g)$ performance.

The package is validated against 12 Gaussian mixture benchmarks with known critical bandwidths stored in `critband/benchmark.py`. Each benchmark case specifies the mixture components (mean, standard deviation, weight per component) and the analytically expected $h_{\text{crit}}$. The benchmarks cover well-separated, moderate, and barely separated bimodal structures; unequal variance and unequal weight conditions; trimodal distributions; skewed, heavy-tailed, and near-unimodal configurations; small-sample regimes ($n=60$); and overlapping variance structures. All benchmarks use a fixed seed (42) for full reproducibility, with mean absolute relative error below 0.5% across all cases.

The package includes 274 unit tests and a technical test report that documents approximately 60% overall line coverage, with the core `bandwidth.py` module at about 45% and the I/O modules at 95--100%. Tests cover core algorithm correctness (Silverman bandwidth, KDE evaluation with both direct and FFT methods, mode counting, critical bandwidth search at multiple tolerances with all three solver methods, trough detection with and without quadratic refinement, component decomposition, $k$-mode detection), I/O adapters (all nine format parsers with read and error paths), and bootstrap inference with reproducible random state. Phase 2 additions include tests for `find_modes` (arbitrary-$k$ detection), `bimodality_strength` (interpretable dip-ratio-based metric), `excess_mass` (excess mass statistic with bootstrap calibration), `dip_test` (Hartigan's dip test with bootstrap $p$-value), `silverman_test` (Silverman's bootstrap test for $k$ modes), and numerical stability guards for constant or near-constant data.

Bootstrap inference is integrated into the main `critical_bandwidth()` call via the `return_ci=True` parameter, which resamples the data with replacement, recomputes $h_{\text{crit}}$ for each resample, and returns percentile confidence intervals and standard error alongside the point estimate. A client-side web deployment using Pyodide (WebAssembly) allows users to run critband entirely in the browser at https://qhwangantoneva.github.io/critband/ — data never leaves the user's machine.

All dependencies are pure Python. The core algorithm requires only `numpy` and `scipy`. The I/O subsystem adds `openpyxl`, `python-docx`, `xlrd`, and `pdfplumber` for format support. The package is installable via `pip install critband` and fully compatible with Python 3.10 and later.

# References

Silverman (1981) introduced the critical bandwidth test for multimodality using kernel density estimates [@silverman1981]. Hartigan & Hartigan (1985) proposed the dip test of unimodality as a complementary nonparametric approach [@hartigan1985]. The package and its validation are described in detail in the arXiv preprint at https://arxiv.org/abs/2605.18686.
