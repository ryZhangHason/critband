---
title: "pola: Critical Bandwidth Analysis for Bimodal Distributions in Python"
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

**pola** is a Python package for detecting and analyzing bimodal distributions using the critical bandwidth method in kernel density estimation (KDE). The critical bandwidth $h_{\text{crit}}$ is the smallest bandwidth at which the KDE transitions from bimodal to unimodal — a well-established statistical test introduced by Silverman (1981). The package provides a single-call interface: `critical_bandwidth(x) → (h_crit, success)`, returning the exact transition point and a convergence flag. pola is the first Python package dedicated to this method. It also offers bimodal decomposition into Gaussian components (`detect_components`), bootstrap confidence intervals (`bootstrap_critical_bandwidth`), and a multi-format I/O subsystem (`pola.io`) that reads numerical data from nine file formats — CSV, TSV, JSON, Markdown, HTML, XLSX, XLS, DOCX, and PDF — through a unified API. The package is designed for researchers and practitioners who need a principled, reproducible test for bimodality without leaving the Python ecosystem.

# Statement of Need

Detecting whether a univariate distribution is meaningfully bimodal is a recurring question across ecology, economics, genomics, and astronomy. Silverman's critical bandwidth test (1981) provides a principled answer, but no dedicated Python implementation has existed until now. Existing tools are scattered across the R ecosystem — the `multimode` package implements Silverman's test and the excess mass test, `diptest` implements Hartigan's dip test, and `ks` provides kernel smoothing with modality testing. None of these are accessible to Python users without running a separate R environment or using subprocess bridges.

The dip test (Hartigan & Hartigan, 1985) is a complementary nonparametric approach but measures departure from unimodality in the cumulative distribution function rather than the density, and is less sensitive to certain bimodal structures. pola fills this gap with a pure-Python implementation of the critical bandwidth method that includes: (1) binary search on KDE mode counts with automatic bounds from Silverman's rule, (2) a coarse-to-fine search strategy for robust convergence, (3) bootstrap percentile confidence intervals for uncertainty quantification, (4) bimodal component decomposition with trough detection, and (5) transparent multi-format data loading that eliminates a common friction point in applied analysis. The package requires only pure Python dependencies — `numpy`, `scipy`, `openpyxl`, `python-docx`, `xlrd`, and `pdfplumber` — and installs with a single `pip install pola` command.

# Implementation and Validation

The core algorithm finds the critical bandwidth via binary search on KDE mode counts. Given input data $x$, the solver first computes Silverman's rule-of-thumb bandwidth $h_s = 1.06 \cdot \min(\hat{\sigma}, \text{IQR}/1.34) \cdot n^{-1/5}$ and sets the search interval to $[h_s / 20, 10 \cdot h_s]$, which empirically brackets the transition for typical bimodal distributions. In the default `"auto"` mode, a coarse binary search (10 iterations) narrows the bracket before a full binary search refines the result to user-specified tolerance (default $10^{-6}$). Mode counting uses `scipy.signal.find_peaks` with a prominence threshold to filter spurious modes from numerical noise. At each iteration, the KDE is evaluated on an adaptive grid whose resolution scales with data size: $\min(800, \max(200, n/10))$ points.

The package is validated against seven Gaussian mixture benchmarks with known critical bandwidths stored in `pola/benchmark.py`. Each benchmark specifies the two components (mean, standard deviation, weight) and the analytically expected $h_{\text{crit}}$ computed from the transition where the bimodal structure disappears. The solver is tested across 10 random seeds per benchmark to assess stability under Monte Carlo variation, with mean absolute relative error below 0.5% across all benchmarks.

The package includes 160 unit tests achieving 97% line coverage. Tests cover core algorithm correctness (Silverman bandwidth, KDE evaluation, mode counting, critical bandwidth search at multiple tolerances, trough detection with and without quadratic refinement, component decomposition), I/O adapters (all nine format parsers with read and error paths), and bootstrap inference with reproducible random state.

Bootstrap inference is implemented via `bootstrap_critical_bandwidth(x, n_resamples=999)`, which resamples the data with replacement, recomputes $h_{\text{crit}}$ for each resample, and returns a `BootstrapResult` dataclass with percentile confidence intervals and standard error. A client-side web deployment using Pyodide (WebAssembly) allows users to run pola entirely in the browser at https://qhwangantoneva.github.io/pola/ — data never leaves the user's machine.

All dependencies are pure Python. The core algorithm requires only `numpy` and `scipy`. The I/O subsystem adds `openpyxl`, `python-docx`, `xlrd`, and `pdfplumber` for format support. The package is installable via `pip install pola` and fully compatible with Python 3.10 and later.

# References

Silverman (1981) introduced the critical bandwidth test for multimodality using kernel density estimates [@silverman1981]. Hartigan & Hartigan (1985) proposed the dip test of unimodality as a complementary nonparametric approach [@hartigan1985]. The package and its validation are described in detail in the arXiv preprint at https://arxiv.org/abs/XXXX.XXXXX.
