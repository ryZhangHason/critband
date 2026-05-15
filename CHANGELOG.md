# Changelog

All notable changes to **pola** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-15

### Added
- `paper.md` — JOSS paper draft with abstract, statement of need, and references
- `CONTRIBUTING.md` — contribution guidelines for community contributors
- `CHANGELOG.md` — this changelog file
- `CITATION.cff` — citation metadata for academic referencing
- 5 new benchmark cases in `BENCHMARK_CASES`:
  - `skewed_bimodal` — asymmetric mixture with unequal variances
  - `heavy_tailed_bimodal` — wide-variance bimodal distribution
  - `near_unimodal` — weakly bimodal with nearly merged modes
  - `small_sample_bimodal` — bimodal with only 30 points per component
  - `overlapping_variances` — overlapping modes with different variances

### Changed
- Bumped version to 0.1.0 for JOSS submission readiness
- Updated `pyproject.toml` — version, authors, classifiers

### Fixed
- Removed dead `method="brent"` code path from `critical_bandwidth` (JOSS #3)
- Removed obsolete `test_brent_matches_binary` test
- Renamed `test_auto_fallback_on_brent_failure` to `test_auto_constant_data_no_crash`
- Updated benchmark validation and stability tests

## [0.0.3] - 2026-05-14

### Added
- Web deployment infrastructure: Pyodide-based frontend on GitHub Pages
- `pola.io` input adapter subpackage — 9 format parsers (CSV, TSV, JSON, Markdown, HTML, XLSX, XLS, DOCX, PDF)
- Bootstrap confidence interval for critical bandwidth (`pola/bootstrap.py`)
- Custom kernel support threaded through the entire call chain
- API reference documentation in docstrings
- Methodology section in README with theoretical background
- OpenWolf context management integration
- Performance benchmark and bandwidth demo examples
- Workflow automation: ruff lint, pre-commit hooks, release workflow
- CI/CD: automated PyPI release on tagged commits

### Changed
- Renamed package from `critical-bandwidth` to `pola`
- Upgraded core algorithm: hybrid binary search method (`method="auto"`)
- Enhanced `find_trough` with quadratic interpolation for sub-grid precision
- Added `detect_components` for bimodal decomposition
- Extended IO adapter test coverage to 97%
- Updated project roadmap and task structure

### Fixed
- `count_modes` off-by-one error (returned `sign_changes + 1`, should be `sign_changes`)
- Input validation hardened: NaN/Inf checking, empty data handling
- KDE vectorized for performance

## [0.0.2] - 2026-05-13

### Added
- PyPI package publication
- Tests: 56 bandwidth tests + 44 IO adapter tests
- `silverman_bandwidth`, `critical_bandwidth`, `gaussian_kde` core functions
- `find_trough` with initial peak detection
- 7 benchmark cases with known critical bandwidth values
- Native format input adapters (CSV, TXT, JSON, Markdown, HTML)

### Changed
- Renamed package from `critical-bandwidth` to `pola` for PyPI
- Refactored code structure for readability

### Fixed
- `count_modes` off-by-one correction

## [0.0.1] - 2026-05-08

### Added
- Initial project structure
- Basic KDE and bandwidth calculation functions
- Project documentation, roadmap, and task structure
- Team workspace configuration
