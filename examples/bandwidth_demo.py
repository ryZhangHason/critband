"""
Bandwidth Demo — Visualizing KDE, critical bandwidth, and component detection.

Generates 3 matplotlib figures showing the core pola algorithms.

Usage:
    uv sync --group viz
    uv run python examples/bandwidth_demo.py

Figures are saved to examples/ as PNG files.
"""

import sys
from pathlib import Path

# Check matplotlib availability before importing anything heavy
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend, works without display
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is required. Install with: uv sync --group viz")
    sys.exit(0)

import numpy as np
from scipy import stats as scipy_stats

from pola import (
    critical_bandwidth,
    detect_components,
    find_trough,
    gaussian_kde,
    silverman_bandwidth,
)
from pola.bandwidth import count_modes
from pola.benchmark import BENCHMARK_CASES

HERE = Path(__file__).parent


# ── helpers ──────────────────────────────────────────────────────────────────


def _peak_trough_marks(x, h, grid, kde_vals, prominence=0.01):
    """Return (peak_x, trough_x) for the two highest KDE peaks."""
    from scipy import signal

    peaks = signal.find_peaks(kde_vals, prominence=prominence * np.max(kde_vals))[0]
    if len(peaks) < 2:
        return None, None
    top_two = np.argsort(kde_vals[peaks])[-2:]
    pi = np.sort(peaks[top_two])
    min_idx = np.argmin(kde_vals[pi[0] : pi[1] + 1]) + pi[0]
    return grid[pi], grid[min_idx : min_idx + 1]


def _gaussian_pdf(x, mu, sigma, weight):
    """Weighted Gaussian PDF evaluated at x."""
    return weight * scipy_stats.norm.pdf(x, mu, sigma)


# ── figure 1: bandwidth sweep ───────────────────────────────────────────────


def plot_bandwidth_sweep(x, h_crit):
    """2×2 subplots showing KDE at 4 different bandwidths."""
    h_silver = silverman_bandwidth(x)
    h_values = [0.05, h_silver, h_crit, h_crit * 3]
    labels = [
        "h = 0.05 (too small)",
        f"h = {h_silver:.4f} (Silverman)",
        f"h = {h_crit:.4f} (critical)",
        f"h = {h_crit * 3:.4f} (too large)",
    ]

    grid = np.linspace(x.min() - 1, x.max() + 1, 1000)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("KDE Bandwidth Sweep", fontsize=14, y=1.02)

    for ax, h, label in zip(axes.flat, h_values, labels):
        kde = gaussian_kde(x, grid, h)
        n_modes = count_modes(x, h)

        ax.hist(x, bins=30, density=True, alpha=0.3, color="gray")
        ax.plot(grid, kde, "b-", lw=2)
        px, tx = _peak_trough_marks(x, h, grid, kde)
        if px is not None:
            ax.plot(px, gaussian_kde(x, px, h), "r^", ms=8, label="peaks")
        if tx is not None:
            ax.plot(tx, gaussian_kde(x, tx, h), "go", ms=6, label="trough")
        ax.set_title(f"{label}  |  {n_modes} peak(s)")
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        if px is not None or tx is not None:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = HERE / "kde_bandwidth_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ── figure 2: critical bandwidth transition ─────────────────────────────────


def plot_critical_transition(x, h_crit):
    """Single plot with KDE at bimodal / critical / unimodal bandwidths."""
    h_bimodal = h_crit * 0.85
    h_unimodal = h_crit * 1.15

    grid = np.linspace(x.min() - 1, x.max() + 1, 1000)
    kde_bi = gaussian_kde(x, grid, h_bimodal)
    kde_crit = gaussian_kde(x, grid, h_crit)
    kde_uni = gaussian_kde(x, grid, h_unimodal)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid, kde_bi, "b-", lw=2, label=f"Bimodal  (h = {h_bimodal:.4f})")
    ax.plot(grid, kde_crit, "g-", lw=2, label=f"Critical (h = {h_crit:.4f})")
    ax.plot(grid, kde_uni, "r-", lw=2, label=f"Unimodal (h = {h_unimodal:.4f})")

    # Trough marker on bimodal curve
    trough = find_trough(x, h_bimodal, refine=True)
    if trough is not None:
        y_trough = gaussian_kde(x, np.array([trough]), h_bimodal)[0]
        ax.axvline(trough, color="blue", ls="--", lw=1, alpha=0.6)
        ax.plot(trough, y_trough, "bo", ms=8)

    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Critical Bandwidth Transition", fontsize=13)
    ax.legend(fontsize=10)

    # Annotate critical bandwidth
    ax.annotate(
        f"h_crit = {h_crit:.4f}",
        xy=(0, 0),
        xycoords="axes fraction",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    path = HERE / "critical_bandwidth_transition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ── figure 3: component decomposition ───────────────────────────────────────


def plot_component_decomposition(x, h_factor=0.85):
    """Single plot showing bimodal component detection results."""
    result = detect_components(x, h_factor=h_factor)
    h = result.critical_bandwidth * h_factor

    grid = np.linspace(x.min() - 1, x.max() + 1, 1000)
    kde = gaussian_kde(x, grid, h)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x, bins=30, density=True, alpha=0.3, color="gray")
    ax.plot(grid, kde, "b-", lw=2.5, label="KDE")

    # Component Gaussians
    c1, c2 = result.component1, result.component2
    pdf1 = _gaussian_pdf(grid, c1.mean, c1.std, c1.weight)
    pdf2 = _gaussian_pdf(grid, c2.mean, c2.std, c2.weight)

    ax.plot(
        grid,
        pdf1,
        "orange",
        ls="--",
        lw=2,
        label=f"C1: μ={c1.mean:.3f}, σ={c1.std:.3f}, w={c1.weight:.3f}",
    )
    ax.plot(
        grid,
        pdf2,
        "green",
        ls="--",
        lw=2,
        label=f"C2: μ={c2.mean:.3f}, σ={c2.std:.3f}, w={c2.weight:.3f}",
    )

    # Separation point
    ax.axvline(
        result.separation_point,
        color="red",
        ls=":",
        lw=2,
        alpha=0.7,
        label=f"Split: x={result.separation_point:.4f}",
    )

    # Peak markers
    from scipy import signal

    peaks = signal.find_peaks(kde, prominence=0.01 * np.max(kde))[0]
    if len(peaks) >= 2:
        top_two = np.argsort(kde[peaks])[-2:]
        peak_indices = np.sort(peaks[top_two])
        ax.plot(grid[peak_indices], kde[peak_indices], "r^", ms=10)

    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Bimodal Component Decomposition", fontsize=13)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = HERE / "component_decomposition.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    case = BENCHMARK_CASES["well_separated_equal_var"]
    x = case.generator(42)
    h_silver = silverman_bandwidth(x)
    h_crit, ok = critical_bandwidth(x)

    print("=== Bandwidth Demo ===")
    print(f"Data: {len(x)} points from {case.name}")
    print(f"Silverman bandwidth:  {h_silver:.4f}")
    print(f"Critical bandwidth:   {h_crit:.4f}")

    if not ok:
        print("  (critical bandwidth search did not fully converge)")

    # Trough at sub-critical bandwidth
    trough = find_trough(x, h_crit * 0.85, refine=True)
    if trough is not None:
        print(f"Trough position:      {trough:.4f}")

    # Component decomposition
    result = detect_components(x)
    print(
        f"Component 1: mean={result.component1.mean:+.3f}, "
        f"std={result.component1.std:.3f}, "
        f"weight={result.component1.weight:.3f}"
    )
    print(
        f"Component 2: mean={result.component2.mean:+.3f}, "
        f"std={result.component2.std:.3f}, "
        f"weight={result.component2.weight:.3f}"
    )
    print(f"Dip ratio: {result.dip_ratio:.4f}")
    print()

    print("Figures saved:")
    plot_bandwidth_sweep(x, h_crit)
    plot_critical_transition(x, h_crit)
    plot_component_decomposition(x)
    print()


if __name__ == "__main__":
    main()
