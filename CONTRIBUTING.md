# Contributing to critband

Thank you for your interest in contributing to **critband** — the Python package for critical bandwidth analysis of bimodal distributions.

## How to Report Issues

If you encounter a bug, have a feature request, or have a question about usage, please open a [GitHub Issue](https://github.com/ryZhangHason/Polarization-CBW/issues). When reporting a bug, include:

- A minimal reproducible example (code + data)
- The expected behavior and the actual behavior
- Your environment (Python version, OS, package versions)

## How to Contribute Code

We welcome contributions via the standard fork-and-pull-request workflow:

1. Fork the repository on GitHub
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Ensure all tests pass (see "Running Tests" below)
5. Commit your changes with a descriptive commit message
6. Push to your fork and open a pull request against the `main` branch

Please keep pull requests focused on a single concern. If your change touches multiple concerns (e.g., a bug fix plus a new feature), split them into separate PRs.

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management. To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/ryZhangHason/Polarization-CBW.git
cd Polarization-CBW

# Install runtime dependencies
uv sync

# Install dev dependencies (ruff, pre-commit, pytest, etc.)
uv sync --group dev
```

## Running Tests

Run the full test suite with verbose output:

```bash
uv run python -m pytest tests/ -v
```

Run with coverage reporting:

```bash
uv run python -m pytest tests/ --cov
```

Run a specific test file:

```bash
uv run python -m pytest tests/test_bandwidth.py -v
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting.

### Linting

```bash
uv run ruff check .
```

### Formatting

```bash
uv run ruff format .
```

The project configuration (in `pyproject.toml`) uses:
- Target Python version: 3.10
- Line length: 100 characters
- Quote style: double quotes
- Enabled rules: E, F, I, N, W

## Pre-commit Hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml`. To install them:

```bash
uv run pre-commit install
```

To run all hooks manually:

```bash
uv run pre-commit run --all-files
```

A fallback `.githooks/pre-commit` hook is available for offline use. Install it with:

```bash
git config core.hooksPath .githooks
```

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0](LICENSE) license.
