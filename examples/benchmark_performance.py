"""
Performance Benchmark — Measure execution time of core pola functions.

Measures 6 core functions at 3 data sizes (100, 1000, 10000 points)
using time.perf_counter with multiple runs and median aggregation.

Usage:
    uv run python examples/benchmark_performance.py
    uv run python examples/benchmark_performance.py --runs 10 --sizes 100 500 2000
    uv run python examples/benchmark_performance.py --output results.csv
"""

import argparse
import csv
import statistics
import time
from datetime import date
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# ── data generation ──────────────────────────────────────────────────────────


def _generate_bimodal(n: int, seed: int = 42) -> np.ndarray:
    """Generate n points from a well-separated bimodal Gaussian mixture."""
    rng = np.random.default_rng(seed)
    half = n // 2
    left = rng.normal(-2.0, 0.3, half)
    right = rng.normal(2.0, 0.3, n - half)
    return np.concatenate([left, right])


def _generate_data(sizes: List[int]) -> Dict[int, np.ndarray]:
    """Generate benchmark data for each requested size."""
    return {n: _generate_bimodal(n) for n in sizes}


# ── timing ───────────────────────────────────────────────────────────────────


def _time_func(func: Callable, args: Tuple[Any, ...], num_runs: int) -> float:
    """Run func(*args) repeatedly and return median time in milliseconds."""
    timings = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)  # seconds → ms
    return statistics.median(timings)


# ── benchmark ────────────────────────────────────────────────────────────────


def run_benchmark(
    sizes: List[int],
    runs_light: int = 20,
    runs_heavy: int = 5,
) -> Dict[str, Dict[int, float]]:
    """
    Run all benchmarks and return {func_name: {n: timing_ms}}.

    Parameters
    ----------
    sizes : list of int
        Data sizes to benchmark (e.g. [100, 1000, 10000]).
    runs_light : int
        Number of timing runs for fast functions.
    runs_heavy : int
        Number of timing runs for slow functions (critical_bandwidth etc.).
    """
    # Lazy import to avoid load cost in timing
    from pola import (
        critical_bandwidth,
        detect_components,
        find_trough,
        gaussian_kde,
        silverman_bandwidth,
    )
    from pola.bandwidth import count_modes

    data = _generate_data(sizes)
    results: Dict[str, Dict[int, float]] = {}

    for n in sizes:
        x = data[n]

        # Compute reference values needed by some benchmarks
        h_silver = silverman_bandwidth(x)
        grid = np.linspace(x.min() - 1, x.max() + 1, 1000)

        # ── silverman_bandwidth ──
        t = _time_func(silverman_bandwidth, (x,), runs_light)
        results.setdefault("silverman_bandwidth", {})[n] = t

        # ── gaussian_kde ──
        t = _time_func(gaussian_kde, (x, grid, h_silver), runs_light)
        results.setdefault("gaussian_kde", {})[n] = t

        # ── count_modes ──
        t = _time_func(count_modes, (x, h_silver), runs_light)
        results.setdefault("count_modes", {})[n] = t

        # ── critical_bandwidth ──
        t = _time_func(critical_bandwidth, (x,), runs_heavy)
        results.setdefault("critical_bandwidth", {})[n] = t

        # ── find_trough ──
        t = _time_func(find_trough, (x, h_silver * 0.85), runs_light)
        results.setdefault("find_trough", {})[n] = t

        # ── detect_components ──
        t = _time_func(detect_components, (x,), runs_heavy)
        results.setdefault("detect_components", {})[n] = t

    return results


# ── output formatting ────────────────────────────────────────────────────────


FUNC_LABELS = {
    "silverman_bandwidth": "silverman_bandwidth",
    "gaussian_kde": "gaussian_kde",
    "count_modes": "count_modes",
    "critical_bandwidth": "critical_bandwidth",
    "find_trough": "find_trough",
    "detect_components": "detect_components",
}

FUNC_ORDER = [
    "silverman_bandwidth",
    "gaussian_kde",
    "count_modes",
    "critical_bandwidth",
    "find_trough",
    "detect_components",
]


def _fmt_ms(t: float) -> str:
    """Format milliseconds to a 4-char width string."""
    if t < 0.01:
        return "<0.01"
    return f"{t:.2f}"


def print_table(results: Dict[str, Dict[int, float]], sizes: List[int]) -> None:
    """Print benchmark results as a formatted console table."""
    today = date.today().isoformat()

    print()
    print("=== pola Performance Benchmark ===")
    print(f"Date: {today}")
    size_str = ", ".join(f"n={s}" for s in sizes)
    print(f"Data size: {size_str}")
    print()

    # Column widths
    name_width = 24
    col_width = 10

    # Header
    header = f"{'Function':<{name_width}}"
    for s in sizes:
        header += f"{f'n={s}':>{col_width}}"
    print(header)
    print("-" * (name_width + col_width * len(sizes)))

    # Rows
    for func_name in FUNC_ORDER:
        row = f"{FUNC_LABELS[func_name]:<{name_width}}"
        for s in sizes:
            t = results.get(func_name, {}).get(s, float("nan"))
            row += f"{_fmt_ms(t):>{col_width}}"
        print(row)

    print()
    print("Timing: median of multiple runs, in milliseconds")
    print()


def save_csv(results: Dict[str, Dict[int, float]], sizes: List[int], path: str) -> None:
    """Save benchmark results to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["function"] + [f"n={s}" for s in sizes])
        for func_name in FUNC_ORDER:
            row = [func_name]
            for s in sizes:
                row.append(f"{results.get(func_name, {}).get(s, float('nan')):.4f}")
            writer.writerow(row)
    print(f"Results saved to: {path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark performance of pola core functions.")
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Number of timing runs for all functions (0 = auto: 20 for fast, 5 for slow)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="Data sizes to benchmark (e.g. 100 1000 10000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV file path to save results",
    )
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> None:
    args = parse_args(argv)

    sizes = sorted(args.sizes)
    if args.runs > 0:
        runs_light = args.runs
        runs_heavy = args.runs
    else:
        runs_light = 20
        runs_heavy = 5

    results = run_benchmark(sizes, runs_light=runs_light, runs_heavy=runs_heavy)
    print_table(results, sizes)

    if args.output:
        save_csv(results, sizes, args.output)


if __name__ == "__main__":
    main()
