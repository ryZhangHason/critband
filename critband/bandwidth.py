"""
Bandwidth calculation functions for kernel density estimation.
Includes Silverman's rule of thumb and critical bandwidth detection for bimodal distributions.
"""

from dataclasses import dataclass, field
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy import signal

# Type alias for critical_bandwidth return types
CriticalBandwidthReturn = Union[
    Tuple[float, bool],
    Tuple[float, bool, float, float, float],
]

# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

_KERNELS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "gaussian": lambda u: np.exp(-(u**2) / 2) / np.sqrt(2 * np.pi),
    "epanechnikov": lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1),
    "uniform": lambda u: 0.5 * (np.abs(u) <= 1).astype(float),
    "triangular": lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
}


def _resolve_kernel(kernel: Union[str, Callable]) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve kernel parameter to a callable kernel function."""
    if callable(kernel):
        return kernel
    if kernel not in _KERNELS:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Built-in kernels: {', '.join(sorted(_KERNELS))}"
        )
    return _KERNELS[kernel]


def _validate_input(x: np.ndarray) -> None:
    """Validate input array: must be 1D, non-empty, finite."""
    if x.ndim != 1:
        raise ValueError(f"Input must be 1-dimensional, got {x.ndim} dimensions")
    if len(x) == 0:
        raise ValueError("Input must not be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input contains NaN or infinite values")


def _kde_grid_points(
    n: int,
    h: Optional[float] = None,
    data_range: Optional[float] = None,
    min_grid: int = 200,
    max_grid: Optional[int] = None,
) -> int:
    """Adapt KDE grid resolution to data size and bandwidth.

    Larger datasets need more grid points to resolve fine structure.
    Smaller bandwidths need finer grids to resolve narrow features.

    Parameters
    ----------
    n : int
        Number of data points.
    h : float, optional
        Bandwidth value for scaling.
    data_range : float, optional
        Range of data (max - min) for scaling.
    min_grid : int, optional
        Minimum number of grid points (default 200).
    max_grid : int or None, optional
        Maximum number of grid points. If None, auto-computed based on n
        as max(800, min(5000, n // 2)).

    Returns
    -------
    int
        Number of grid points, clamped to [min_grid, max_grid].
    """
    # Auto-compute max_grid based on data size
    if max_grid is None:
        auto_max = max(800, min(5000, n // 2))
    else:
        auto_max = max_grid

    # Base from data size
    points = min(auto_max, max(min_grid, n // 10))

    # Scale by bandwidth: when h is small relative to range, need finer grid
    if h is not None and data_range is not None and h > 1e-12 and data_range > 0:
        ratio = data_range / (h * 4)
        # ratio > 1 means h is small relative to range -> need more points
        # ratio < 1 means h is large -> can use fewer points
        scale = min(3.0, max(0.5, ratio))
        points = int(points * scale)

    return min(auto_max, max(min_grid, points))


def silverman_bandwidth(x: np.ndarray) -> float:
    """
    Calculate Silverman's rule of thumb bandwidth for univariate data.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data points

    Returns
    -------
    float
        Silverman's rule of thumb bandwidth value

    Notes
    -----
    Formula: h = 1.06 * min(std, IQR/1.34) * n^(-1/5)

    For constant data (std=0, IQR=0), returns a floored value of 0.01
    to avoid degenerate bandwidth of zero.
    """
    _validate_input(x)
    n = len(x)

    if n == 1:
        return 1.0  # Default bandwidth for single point

    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 1.06 * np.min([std, iqr / 1.34]) * (n ** (-0.2))
    return h if h > 0 else 0.01  # floor at 0.01 for constant data


@dataclass
class Mode:
    """A single detected mode (peak) in a kernel density estimate.

    Parameters
    ----------
    position : float
        x-coordinate of the mode peak.
    height : float
        KDE density value at the peak.
    width : float
        Full width at half maximum (FWHM) approximation in x-coordinate units.
    prominence : float
        Prominence of the peak relative to surrounding minima.
    left_base : float
        x-coordinate at the left base of the peak.
    right_base : float
        x-coordinate at the right base of the peak.
    """

    position: float
    height: float
    width: float
    prominence: float
    left_base: float
    right_base: float


@dataclass
class ModeResult:
    """Result of find_modes(): all detected modes at a given bandwidth.

    Parameters
    ----------
    n_modes : int
        Number of detected modes.
    modes : list[Mode]
        List of Mode dataclasses with detailed metadata for each peak.
    bandwidth : float
        Bandwidth at which the KDE was computed.
    grid_points : int
        Number of grid points used for the KDE evaluation.
    """

    n_modes: int
    modes: list = field(default_factory=list)
    bandwidth: float = 0.0
    grid_points: int = 0


def find_modes(
    x: np.ndarray,
    h: float,
    grid_points: Optional[int] = None,
    prominence: float = 0.01,
    kernel: Union[str, Callable] = "gaussian",
) -> ModeResult:
    """Find all modes (peaks) in KDE at bandwidth h with detailed metadata.

    Uses scipy.signal.find_peaks with a prominence threshold to detect modes,
    and scipy.signal.peak_widths to compute full-width at half-maximum (FWHM)
    and peak base positions.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    h : float
        Bandwidth value.
    grid_points : int, optional
        Number of evaluation points. Defaults to adaptive based on data size.
    prominence : float, optional
        Minimum prominence as a fraction of max density (default 0.01).
        Peaks below this fraction of the maximum KDE height are ignored.
    kernel : str or callable, optional
        Kernel function (default "gaussian").

    Returns
    -------
    ModeResult
        A dataclass containing the number of modes and detailed metadata
        for each detected mode.
    """
    _validate_input(x)
    if h <= 0:
        return ModeResult(n_modes=0, modes=[], bandwidth=h, grid_points=0)

    if grid_points is None:
        data_range = x.max() - x.min() if len(x) > 1 else 1.0
        grid_points = _kde_grid_points(len(x), h=h, data_range=data_range)
    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid_points)
    kde_vals = gaussian_kde(x, grid, h, kernel=kernel)

    peaks, properties = signal.find_peaks(kde_vals, prominence=prominence * np.max(kde_vals))

    if len(peaks) == 0:
        return ModeResult(n_modes=0, modes=[], bandwidth=h, grid_points=grid_points)

    # Compute peak widths at half max (FWHM)
    widths, _width_heights, left_ips, right_ips = signal.peak_widths(
        kde_vals, peaks, rel_height=0.5
    )

    dx = grid[1] - grid[0]
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    modes = []
    for i, p in enumerate(peaks):
        # Convert from index units to x-coordinate units
        width_val = float(widths[i] * dx)
        left_base_val = float(max(grid[0], grid[0] + left_ips[i] * dx))
        right_base_val = float(min(grid[-1], grid[0] + right_ips[i] * dx))

        # Guard against NaN from scipy.peak_widths (can happen at edges)
        if np.isnan(width_val):
            width_val = 0.0
        if np.isnan(left_base_val):
            left_base_val = float(grid[0])
        if np.isnan(right_base_val):
            right_base_val = float(grid[-1])

        modes.append(
            Mode(
                position=float(grid[p]),
                height=float(kde_vals[p]),
                width=width_val,
                prominence=float(prominences[i]),
                left_base=left_base_val,
                right_base=right_base_val,
            )
        )

    return ModeResult(
        n_modes=len(modes),
        modes=modes,
        bandwidth=h,
        grid_points=grid_points,
    )


def count_modes(
    x: np.ndarray,
    h: float,
    grid_points: Optional[int] = None,
    prominence: float = 0.01,
    kernel: Union[str, Callable] = "gaussian",
) -> int:
    """
    Count number of modes in kernel density estimate for given bandwidth.

    Delegates to find_modes() internally and returns the mode count.
    The signature is unchanged for backward compatibility.

    Parameters
    ----------
    x : np.ndarray
        Input data
    h : float
        Bandwidth value
    grid_points : int, optional
        Number of evaluation points. Defaults to adaptive based on data size.
    prominence : float, optional
        Minimum prominence as a fraction of max density (default 0.01).
        Peaks below this fraction of the maximum KDE height are ignored.

    Returns
    -------
    int
        Number of detected modes
    """
    result = find_modes(x, h, grid_points=grid_points, prominence=prominence, kernel=kernel)
    return result.n_modes


def _trough_ratio(
    x: np.ndarray,
    h: float,
    grid_points: Optional[int] = None,
    prominence: float = 0.01,
    kernel: Union[str, Callable] = "gaussian",
) -> float:
    """
    Continuous measure of bimodality strength at bandwidth h.

    Returns dip_ratio = min_valley / min_peak_height, a value in (0, 1].

    - dip_ratio near 0: deep trough between modes (strongly bimodal)
    - dip_ratio near 1: no meaningful trough (unimodal)

    This is a descriptive bimodality strength measure, NOT a solver
    objective for critical bandwidth. The critical bandwidth is found
    via binary search on KDE mode counts (see critical_bandwidth).

    For KDE with < 2 peaks, returns 1.0 (unimodal signal).
    """
    _validate_input(x)
    if h <= 0:
        return 1.0

    if grid_points is None:
        data_range = x.max() - x.min() if len(x) > 1 else 1.0
        grid_points = _kde_grid_points(len(x), h=h, data_range=data_range)

    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid_points)
    kde_vals = gaussian_kde(x, grid, h, kernel=kernel)

    peaks = signal.find_peaks(kde_vals, prominence=prominence * np.max(kde_vals))[0]
    if len(peaks) < 2:
        return 1.0

    # Get the two highest peaks
    peak_heights = kde_vals[peaks]
    top_two = np.argsort(peak_heights)[-2:]
    peak_indices = np.sort(peaks[top_two])

    # Minimum KDE value between the two peaks
    left, right = peak_indices[0], peak_indices[1]
    valley_val = np.min(kde_vals[left : right + 1])
    min_peak = min(kde_vals[left], kde_vals[right])

    return valley_val / min_peak if min_peak > 0 else 1.0


def gaussian_kde_fft(
    x: np.ndarray,
    grid: np.ndarray,
    h: float,
    kernel: Union[str, Callable] = "gaussian",
) -> np.ndarray:
    """KDE using FFT convolution for O(g·log(g)) performance.

    For large n (>> grid_points), this is much faster than the direct
    O(n·g) broadcasting method. Accuracy depends on bin width.

    Parameters
    ----------
    x : np.ndarray
        Input data points.
    grid : np.ndarray
        Uniformly spaced evaluation points.
    h : float
        Bandwidth.
    kernel : str or callable, optional
        Kernel function. Built-in options: "gaussian" (default),
        "epanechnikov", "uniform", "triangular".
        Callables must accept an ndarray of standardized distances
        u = (x - xi)/h and return an ndarray of kernel weights.

    Returns
    -------
    np.ndarray
        Density estimates at grid points, normalized to integrate to ~1.
    """
    _validate_input(x)
    dx = grid[1] - grid[0]
    g = len(grid)
    n = len(x)
    kernel_func = _resolve_kernel(kernel)

    # 1. Bin the data onto the grid (histogram)
    bins = np.linspace(grid[0] - dx / 2, grid[-1] + dx / 2, g + 1)
    hist, _ = np.histogram(x, bins=bins)

    # 2. Evaluate kernel at grid points
    kernel_grid = np.linspace(-(g // 2) * dx, (g // 2) * dx, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = kernel_grid / h
        kernel_vals = kernel_func(u)  # raw kernel values

    # 3. FFT convolution (raw kernel, no ifftshift)
    # np.convolve(hist, kernel_vals, mode='same') equivalent
    fft_len = 1
    while fft_len < 2 * g:
        fft_len *= 2
    hist_pad = np.zeros(fft_len)
    hist_pad[:g] = hist
    kernel_pad = np.zeros(fft_len)
    kernel_pad[:g] = kernel_vals

    conv = np.fft.irfft(
        np.fft.rfft(hist_pad) * np.fft.rfft(kernel_pad),
        n=fft_len,
    )

    # 4. Extract mode='same' equivalent: center g values
    start = (g - 1) // 2
    density = conv[start : start + g].copy()

    # 5. Normalize: same as direct method
    density /= n * h

    return density


def gaussian_kde(
    x: np.ndarray,
    grid: np.ndarray,
    h: float,
    kernel: Union[str, Callable] = "gaussian",
    use_fft: Optional[bool] = None,
) -> np.ndarray:
    """
    Kernel density estimation using vectorized broadcasting or FFT convolution.

    Parameters
    ----------
    x : np.ndarray
        Input data points
    grid : np.ndarray
        Points where density is evaluated
    h : float
        Bandwidth
    kernel : str or callable, optional
        Kernel function. Built-in options: "gaussian" (default),
        "epanechnikov", "uniform", "triangular".
        Callables must accept an ndarray of standardized distances
        u = (x - xi)/h and return an ndarray of kernel weights.
    use_fft : bool or None, optional
        If True, use FFT-accelerated method.
        If False, use direct vectorized broadcasting.
        If None (default), auto-select: FFT when n > 5000 and g < n.

    Returns
    -------
    np.ndarray
        Density estimates at grid points
    """
    _validate_input(x)
    n = len(x)
    g = len(grid)

    # Auto-select or explicit FFT
    if use_fft is True or (use_fft is None and n > 5000 and g < n):
        return gaussian_kde_fft(x, grid, h, kernel=kernel)

    # Direct vectorized broadcasting method
    kernel_func = _resolve_kernel(kernel)
    diff = grid[:, np.newaxis] - x[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = diff / h
        density = np.sum(kernel_func(u), axis=1)
    return density / (n * h)


def _bracket_critical(
    x: np.ndarray,
    h_min: float,
    h_max: float,
    max_iter: int = 10,
    kernel: Union[str, Callable] = "gaussian",
    k: int = 2,
) -> Tuple[float, float]:
    """
    Narrow bracket around critical bandwidth using binary search on count_modes.

    Returns (low, high) where count_modes(x, low) >= k and count_modes(x, high) < k.
    Caller must ensure boundary conditions are met before calling.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data
    h_min : float
        Lower bound for bandwidth search
    h_max : float
        Upper bound for bandwidth search
    max_iter : int, optional
        Maximum number of iterations (default 10)
    kernel : str or callable, optional
        Kernel function (default "gaussian")
    k : int, optional
        Number of modes to detect. Default is 2 (bimodal).
        The function finds the bandwidth where KDE transitions from
        >=k modes to <k modes.
    """
    low, high = h_min, h_max
    for _ in range(max_iter):
        mid = (low + high) / 2
        if count_modes(x, mid, kernel=kernel) >= k:
            low = mid
        else:
            high = mid
    return low, high


def _brent_objective(
    h: float,
    x: np.ndarray,
    kernel: Union[str, Callable] = "gaussian",
    k: int = 2,
) -> float:
    """Continuous objective function for Brent's method.

    Returns positive when >=k modes are detected (h < h_crit), negative when
    <k modes are detected (h > h_crit), crossing zero at the critical bandwidth.

    For h where k+ modes are detected, uses `1 - trough_ratio` which approaches 0
    as the valley fills in (h → h_crit⁻). For h with <k modes, returns a constant
    negative value, creating a clean sign change at h_crit.

    Parameters
    ----------
    h : float
        Bandwidth to evaluate
    x : np.ndarray
        1D array of input data
    kernel : str or callable, optional
        Kernel function (default "gaussian")
    k : int, optional
        Number of modes to detect (default 2).
    """
    if h <= 0:
        return 1.0
    n_modes = count_modes(x, h, kernel=kernel)
    if n_modes >= k:
        tr = _trough_ratio(x, h, kernel=kernel)
        # Positive, approaches 0+ as trough fills near h_crit
        return max(1.0 - tr, 1e-10)
    else:
        # <k modes: return negative. A fixed -1.0 creates a clear sign change
        # at h_crit regardless of how far past the transition we are.
        return -1.0


def critical_bandwidth(
    x: np.ndarray,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    method: str = "auto",
    k: int = 2,
    kernel: Union[str, Callable] = "gaussian",
    return_ci: bool = False,
    ci_resamples: int = 999,
    ci_alpha: float = 0.05,
    ci_random_state: Optional[int] = None,
) -> CriticalBandwidthReturn:
    """
    Calculate critical bandwidth for k-mode detection.

    Critical bandwidth is the smallest bandwidth where the kernel density
    estimate has fewer than k modes, found via binary search on KDE mode counts
    or Brent's method on a continuous bimodality objective.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data
    h_min : float, optional
        Lower bound for bandwidth search
    h_max : float, optional
        Upper bound for bandwidth search
    tol : float, optional
        Convergence tolerance (used for binary and Brent fallback)
    max_iter : int, optional
        Maximum number of iterations
    method : str, optional
        Search method:
        - "auto" (default): intelligently selects the optimal method based
          on data characteristics. Uses Brent for large (n≥100), well-separated
          (trough_ratio < 0.7) data, binary search otherwise. Falls back to
          binary search if Brent fails.
        - "binary": pure binary search on KDE mode counts.
        - "brent": pure Brent's method using scipy.optimize.brentq on a
          continuous trough-ratio objective. Falls back to binary search
          if brentq fails to converge or produces an invalid result.
    k : int, optional
        Number of modes to detect (default 2). The function finds the
        bandwidth where KDE transitions from >=k modes to <k modes.
        For k=2 (default), this is the classic bimodal critical bandwidth.
        For k=3, finds the bandwidth where trimodality disappears, etc.
    kernel : str or callable, optional
        Kernel function. Built-in: "gaussian" (default), "epanechnikov",
        "uniform", "triangular". Also accepts callables.
    return_ci : bool, optional
        If True, also return a bootstrap confidence interval (default False).
    ci_resamples : int, optional
        Number of bootstrap resamples for CI (default 999).
    ci_alpha : float, optional
        Significance level for CI (default 0.05 → 95% CI).
    ci_random_state : int, optional
        Random seed for reproducible bootstrap resampling.

    Returns
    -------
    CriticalBandwidthReturn
        When return_ci=False (default): (critical_bandwidth_value, convergence_success)
        When return_ci=True: (critical_bandwidth_value, convergence_success,
            ci_lower, ci_upper, standard_error)

    Notes
    -----
    The "auto" and "binary" methods find h_crit via binary search on KDE
    mode counts. The "brent" method uses a continuous objective based on
    `_trough_ratio`: f(h) = 1 - trough_ratio when >=k modes (> 0), f(h) = -1
    when <k modes (< 0), crossing zero at the transition.

    dip_ratio (via _trough_ratio) is available as a descriptive bimodality
    strength measure (0 = strongly bimodal, 1 = unimodal).

    When return_ci=True, the function also computes a bootstrap confidence interval
    via bootstrap_critical_bandwidth() and returns (h_crit, ok, ci_lower, ci_upper, se).
    Bootstrap resamples that fail to converge are excluded from the CI calculation.
    """
    _validate_input(x)

    if h_min is None:
        h_min = silverman_bandwidth(x) / 20.0
    if h_max is None:
        h_max = silverman_bandwidth(x) * 10.0

    if h_min <= 0:
        h_min = 1e-8

    # Check boundary conditions
    modes_min = count_modes(x, h_min, kernel=kernel)
    modes_max = count_modes(x, h_max, kernel=kernel)

    if modes_max >= k:
        h_crit, ok = h_max, False
    elif modes_min < k:
        h_crit, ok = h_min, False
    elif method == "brent":
        try:
            from scipy.optimize import brentq

            h_crit = brentq(
                _brent_objective,
                h_min,
                h_max,
                args=(x, kernel, k),
                xtol=tol,
                maxiter=max_iter,
            )
            # Verify and adjust: the objective may converge just before the
            # discrete count_modes transition. If still >=k modes, nudge upward
            # in small increments until <k modes, then accept the result.
            h_test = h_crit
            for _ in range(20):
                if count_modes(x, h_test, kernel=kernel) < k:
                    h_crit, ok = h_test, True
                    break
                h_test *= 1.001  # nudge upward toward <k side
            else:
                # Verification failed after nudging, fallback
                h_crit, ok = _binary_search_critical(x, h_min, h_max, tol, max_iter, kernel, k)
        except (ValueError, RuntimeError):
            h_crit, ok = _binary_search_critical(x, h_min, h_max, tol, max_iter, kernel, k)
    elif method == "auto":
        n = len(x)

        # Small sample → binary (Brent is unstable on small n)
        if n < 100:
            low, high = _bracket_critical(
                x, h_min, h_max, min(10, max_iter // 3), kernel=kernel, k=k
            )
            h_crit, ok = _binary_search_critical(x, low, high, tol, max_iter, kernel, k)
        else:
            # Evaluate trough ratio at h_min as separation strength measure.
            tr_min = _trough_ratio(x, h_min, kernel=kernel)

            if tr_min < 0.7:
                try:
                    from scipy.optimize import brentq

                    h_crit = brentq(
                        _brent_objective,
                        h_min,
                        h_max,
                        args=(x, kernel, k),
                        xtol=tol,
                        maxiter=max_iter,
                    )
                    h_test = h_crit
                    for _ in range(20):
                        if count_modes(x, h_test, kernel=kernel) < k:
                            h_crit, ok = h_test, True
                            break
                        h_test *= 1.001
                    else:
                        # Verification failed, fall through
                        low, high = _bracket_critical(
                            x, h_min, h_max, min(10, max_iter // 3), kernel=kernel, k=k
                        )
                        h_crit, ok = _binary_search_critical(x, low, high, tol, max_iter, kernel, k)
                except (ValueError, RuntimeError):
                    low, high = _bracket_critical(
                        x, h_min, h_max, min(10, max_iter // 3), kernel=kernel, k=k
                    )
                    h_crit, ok = _binary_search_critical(x, low, high, tol, max_iter, kernel, k)
            else:
                low, high = _bracket_critical(
                    x, h_min, h_max, min(10, max_iter // 3), kernel=kernel, k=k
                )
                h_crit, ok = _binary_search_critical(x, low, high, tol, max_iter, kernel, k)
    else:
        # Binary search (method="binary")
        low, high = h_min, h_max
        h_crit, ok = _binary_search_critical(x, low, high, tol, max_iter, kernel, k)

    if return_ci:
        from critband.bootstrap import bootstrap_critical_bandwidth

        boot_kwargs = {}
        if h_min is not None:
            boot_kwargs["h_min"] = h_min
        if h_max is not None:
            boot_kwargs["h_max"] = h_max
        boot_kwargs["tol"] = tol
        boot_kwargs["max_iter"] = max_iter
        boot_kwargs["method"] = method
        boot_kwargs["k"] = k
        if kernel != "gaussian":
            boot_kwargs["kernel"] = kernel

        boot = bootstrap_critical_bandwidth(
            x,
            n_resamples=ci_resamples,
            alpha=ci_alpha,
            random_state=ci_random_state,
            **boot_kwargs,
        )
        return h_crit, ok, boot.ci_lower, boot.ci_upper, boot.standard_error

    return h_crit, ok


def _binary_search_critical(
    x: np.ndarray,
    low: float,
    high: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    kernel: Union[str, Callable] = "gaussian",
    k: int = 2,
) -> Tuple[float, bool]:
    """Binary search for critical bandwidth.

    Assumes count_modes(x, low) >= k and count_modes(x, high) < k.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data
    low : float
        Lower bound (must have >=k modes)
    high : float
        Upper bound (must have <k modes)
    tol : float, optional
        Convergence tolerance (default 1e-6)
    max_iter : int, optional
        Maximum number of iterations (default 100)
    kernel : str or callable, optional
        Kernel function (default "gaussian")
    k : int, optional
        Number of modes to detect (default 2).
        The function finds the bandwidth where KDE transitions from
        >=k modes to <k modes.
    """
    for _ in range(max_iter):
        mid = (low + high) / 2
        if count_modes(x, mid, kernel=kernel) >= k:
            low = mid
        else:
            high = mid
        if high - low < tol:
            break
    return (low + high) / 2, True


def find_trough(
    x: np.ndarray,
    h: float,
    grid_points: Optional[int] = None,
    prominence: float = 0.01,
    refine: bool = True,
    kernel: Union[str, Callable] = "gaussian",
) -> Optional[float]:
    """
    Find the trough (lowest point) between the two most prominent KDE modes.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data
    h : float
        Bandwidth at which to evaluate the KDE
    grid_points : int, optional
        Number of evaluation points (default 1000)
    prominence : float, optional
        Minimum prominence as fraction of max density (default 0.01)
    refine : bool, optional
        If True, use quadratic interpolation to refine trough position
    kernel : str or callable, optional
        Kernel function (default "gaussian").

    Returns
    -------
    float or None
        x-coordinate of the trough between the two highest KDE peaks,
        or None if fewer than 2 peaks are detected.
    """
    _validate_input(x)
    if h <= 0:
        return None

    if grid_points is None:
        data_range = x.max() - x.min() if len(x) > 1 else 1.0
        grid_points = _kde_grid_points(len(x), h=h, data_range=data_range)
    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid_points)
    kde_vals = gaussian_kde(x, grid, h, kernel=kernel)

    peaks = signal.find_peaks(kde_vals, prominence=prominence * np.max(kde_vals))[0]
    if len(peaks) < 2:
        return None

    # Two highest peaks
    peak_heights = kde_vals[peaks]
    top_two = np.argsort(peak_heights)[-2:]
    peak_indices = np.sort(peaks[top_two])

    left, right = peak_indices[0], peak_indices[1]
    min_idx = np.argmin(kde_vals[left : right + 1]) + left

    if not refine:
        return grid[min_idx]

    # Quadratic interpolation for sub-grid precision
    i = min_idx
    if 0 < i < len(grid) - 1:
        x0, x1, x2 = grid[i - 1], grid[i], grid[i + 1]
        y0, y1, y2 = kde_vals[i - 1], kde_vals[i], kde_vals[i + 1]
        denom = (x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)
        if abs(denom) > 1e-15:
            return x1 - 0.5 * ((x2 - x1) ** 2 * (y1 - y0) - (x1 - x0) ** 2 * (y2 - y1)) / denom

    return float(grid[min_idx])


@dataclass
class Component:
    """Estimated parameters of a single Gaussian component in a bimodal mixture."""

    mean: float
    std: float
    weight: float


@dataclass
class BimodalDecomposition:
    """Complete decomposition of a bimodal distribution into two components."""

    critical_bandwidth: float
    component1: Component
    component2: Component
    separation_point: float
    dip_ratio: float
    is_exploratory: bool = True
    note: str = ""


def detect_components(
    x: np.ndarray,
    h_factor: float = 0.85,
    grid_points: Optional[int] = None,
    kernel: Union[str, Callable] = "gaussian",
) -> BimodalDecomposition:
    """
    Decompose a bimodal distribution into two component Gaussians.

    Splits data at the KDE trough between the two highest peaks at a
    sub-critical bandwidth, then computes sample statistics for each side.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data
    h_factor : float, optional
        Multiplier on critical bandwidth for analysis (default 0.85).
        Lower = more bimodal (clearer trough), higher = closer to transition.
    grid_points : int, optional
        KDE grid resolution (default 1000)
    kernel : str or callable, optional
        Kernel function (default "gaussian").

    Returns
    -------
    BimodalDecomposition
        Structured exploratory decomposition with component parameters.

    Raises
    ------
    ValueError
        If fewer than 2 KDE peaks are detected at the analysis bandwidth.
    """
    _validate_input(x)

    h_crit, ok = critical_bandwidth(x, kernel=kernel)
    if not ok:
        warnings.warn(
            "Critical bandwidth search did not fully converge; proceeding with best estimate."
        )

    h_analysis = h_crit * h_factor

    if grid_points is None:
        data_range = x.max() - x.min() if len(x) > 1 else 1.0
        grid_points = _kde_grid_points(len(x), h=h_analysis, data_range=data_range)

    # Build KDE and find peaks
    grid = np.linspace(x.min() - 3 * h_analysis, x.max() + 3 * h_analysis, grid_points)
    kde_vals = gaussian_kde(x, grid, h_analysis, kernel=kernel)
    peaks = signal.find_peaks(kde_vals, prominence=0.01 * np.max(kde_vals))[0]

    if len(peaks) < 2:
        raise ValueError(
            f"Data does not appear bimodal: only {len(peaks)} KDE peak(s) "
            f"detected at h={h_analysis:.4f} (h_crit={h_crit:.4f}). "
            "The distribution may be unimodal or very weakly bimodal."
        )

    # Find the two highest peaks and the trough between them
    peak_heights = kde_vals[peaks]
    top_two = np.argsort(peak_heights)[-2:]
    peak_indices = np.sort(peaks[top_two])
    left, right = peak_indices[0], peak_indices[1]
    min_idx = np.argmin(kde_vals[left : right + 1]) + left
    trough_x = float(grid[min_idx])

    valley_val = float(np.min(kde_vals[left : right + 1]))
    min_peak = float(min(kde_vals[left], kde_vals[right]))
    dip_ratio = valley_val / min_peak if min_peak > 0 else 1.0

    # Split data at the trough and compute component statistics
    x_left = x[x <= trough_x]
    x_right = x[x > trough_x]

    if len(x_left) == 0 or len(x_right) == 0:
        raise ValueError(
            "Trough-based split produced an empty component. "
            "The data may be unimodal or the trough is at the data boundary."
        )

    note = ""
    if dip_ratio > 0.70:
        note = "Overlapping or weakly separated mixture; trough split should be treated as exploratory."
        warnings.warn(note)

    n_total = len(x)
    comp1 = Component(
        mean=float(np.mean(x_left)),
        std=float(np.std(x_left, ddof=1)),
        weight=len(x_left) / n_total,
    )
    comp2 = Component(
        mean=float(np.mean(x_right)),
        std=float(np.std(x_right, ddof=1)),
        weight=len(x_right) / n_total,
    )

    # Ensure left = lower mean, right = higher mean
    if comp2.mean < comp1.mean:
        comp1, comp2 = comp2, comp1

    return BimodalDecomposition(
        critical_bandwidth=h_crit,
        component1=comp1,
        component2=comp2,
        separation_point=trough_x,
        dip_ratio=dip_ratio,
        is_exploratory=True,
        note=note,
    )


# ---------------------------------------------------------------------------
# Hartigan's dip test for unimodality
# ---------------------------------------------------------------------------


@dataclass
class DipTestResult:
    """Result of Hartigan's dip test for unimodality.

    Parameters
    ----------
    dip : float
        The dip statistic (larger values indicate greater departure from unimodality).
    p_value : float
        Bootstrap p-value for the test.
    n_boot : int
        Number of bootstrap resamples used.
    n_extreme : int
        Number of bootstrap samples with dip >= observed dip.
    """

    dip: float
    p_value: float
    n_boot: int
    n_extreme: int


def _gcm(x, y):
    """Greatest convex minorant. O(n).

    Computes the convex envelope of points (x[i], y[i]) sorted by x.
    Returns the GCM y-values evaluated at each x[i].

    Uses a pool-adjacent-violators (PAV) algorithm: points are pushed as
    single-point blocks and merged when adjacent blocks violate convexity
    (decreasing slope).
    """
    # A block stores [x_start, x_end, y_start, y_end]
    # The convex function is linear within each block
    stack = []
    n = len(x)
    for i in range(n):
        # Push point as a block of size 1
        stack.append([x[i], x[i], y[i], y[i]])
        # While the last two blocks violate convexity (decreasing slope)
        while len(stack) >= 2:
            x1a, x1b, y1a, y1b = stack[-2]
            x2a, x2b, y2a, y2b = stack[-1]
            # Convexity requires slope1 ≤ slope_total (the slope from start of
            # block 1 to end of block 2). If slope1 > slope_total, combine blocks.
            s1 = (y1b - y1a) / (x1b - x1a) if x1b > x1a else float("inf")
            s_total = (y2b - y1a) / (x2b - x1a)
            if s1 <= s_total:  # convexity satisfied
                break
            # Merge: the combined block goes from (x1a, y1a) to (x2b, y2b)
            stack.pop()
            stack[-1] = [x1a, x2b, y1a, y2b]

    # Interpolate to get GCM y-values at each x[i]
    gcm_y = np.empty(n)
    block_idx = 0
    for i in range(n):
        while block_idx < len(stack) and x[i] > stack[block_idx][1]:
            block_idx += 1
        if block_idx >= len(stack):
            block_idx = len(stack) - 1
        x0, x1, y0, y1 = stack[block_idx]
        if x1 > x0:
            gcm_y[i] = y0 + (y1 - y0) * (x[i] - x0) / (x1 - x0)
        else:
            gcm_y[i] = y0
    return gcm_y


def _lcm(x, y):
    """Least concave majorant. O(n).

    LCM is concave and lies above all points (x[i], y[i]).
    Computed as -GCM(-x_rev, -y_rev) where x_rev = -x[::-1] and
    y_rev = -y[::-1], with the result reversed back.
    """
    # Negate and reverse: x is increasing, -x is decreasing, but _gcm needs increasing x.
    # So we reverse: -x[::-1] is increasing, -y[::-1] reverses too.
    x_rev = -np.asanyarray(x)[::-1]
    y_rev = -np.asanyarray(y)[::-1]
    gcm_y = _gcm(x_rev, y_rev)
    return -np.asanyarray(gcm_y)[::-1]


def _compute_dip_statistic(x):
    """Compute Hartigan's dip statistic for data x.

    The dip statistic D is the maximum vertical distance between the
    empirical CDF and the best-fitting unimodal CDF. Values range from
    0 (perfectly unimodal) to 0.5 (maximum possible departure).

    Uses O(n²) precomputation: GCM for each left tail and LCM for each
    right tail are computed once, then all modal intervals are evaluated
    using vectorized numpy operations for the middle-segment deviation.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data.

    Returns
    -------
    float
        The dip statistic.

    Performance
    -----------
    The algorithm has O(n²) precomputation (GCM/LCM for each prefix/suffix)
    followed by O(n²) interval evaluation with vectorized inner loops.
    For n=400, one call takes ~0.1s; dip_test(n_boot=999) takes ~100s.
    """
    x_sorted = np.sort(x)
    n = len(x_sorted)
    if n < 3:
        return 0.0

    fn = np.arange(1, n + 1) / n  # Empirical CDF

    # Precompute dip_left[a] = max dip on left tail when mode ends at x[a]
    # This is max over i ≤ a of (F[i] - GCM_a(i))
    dip_left = np.zeros(n)
    for a in range(1, n):  # need at least 2 points
        x_left = x_sorted[: a + 1]
        y_left = fn[: a + 1]
        gcm_y = _gcm(x_left, y_left)
        dip_left[a] = np.max(y_left - gcm_y)
    # dip_left[0] stays 0 (only 1 point, trivial fit)

    # Precompute dip_right[b] = max dip on right tail when mode starts at x[b]
    # This is max over i ≥ b of (LCM_b(i) - F[i])
    dip_right = np.zeros(n)
    for b in range(n - 1):
        x_right = x_sorted[b:]
        y_right = fn[b:]
        lcm_y = _lcm(x_right, y_right)
        dip_right[b] = np.max(lcm_y - y_right)
    # dip_right[n-1] stays 0

    # Main loop: for each interval (a, b), compute max deviation from
    # the linear interpolation line, combined with left/right tail dip.
    # Uses vectorized numpy operations for the middle-segment computation.
    best_dip = 0.5

    for a in range(n - 2):
        xa, fa = x_sorted[a], fn[a]
        for b in range(a + 2, n):
            xb, fb = x_sorted[b], fn[b]
            if xb > xa:
                # Vectorized: compute max deviation for all k in (a,b) at once
                xs_seg = x_sorted[a + 1 : b]
                fn_seg = fn[a + 1 : b]
                linear_vals = fa + (fb - fa) * (xs_seg - xa) / (xb - xa)
                mid_dip = float(np.max(np.abs(fn_seg - linear_vals)))
            else:
                mid_dip = 0.0

            dip = dip_left[a]
            if mid_dip > dip:
                dip = mid_dip
            if dip_right[b] > dip:
                dip = dip_right[b]

            if dip < best_dip:
                best_dip = dip

            if best_dip == 0:
                return 0.0

    return best_dip


def dip_test(x, n_boot=999, random_state=None):
    """Hartigan's dip test for unimodality.

    Tests the null hypothesis that the distribution of x is unimodal
    against the alternative that it is not (multimodal).

    Parameters
    ----------
    x : np.ndarray
        1D array of input data.
    n_boot : int, optional
        Number of bootstrap resamples for p-value calibration (default 999).
    random_state : int or None, optional
        Random seed for reproducible bootstrap resampling.

    Returns
    -------
    DipTestResult
        A dataclass with dip statistic, p-value, and bootstrap details.

    Notes
    -----
    The p-value is calibrated by sampling from the uniform distribution
    on [0, 1], which is the least favorable unimodal distribution under
    the null. This is the standard approach used by the R 'diptest' package.
    """
    if len(x) > 5000 and n_boot >= 999:
        warnings.warn(
            "dip_test() is expensive for large samples at the default bootstrap count; "
            "consider a smaller n_boot or a complementary multimodality check."
        )

    # Compute observed dip
    dip_obs = _compute_dip_statistic(x)

    n = len(x)
    rng = np.random.default_rng(random_state)

    # Bootstrap under H0: sample from Uniform(0,1) — the least favorable
    # unimodal distribution (standard approach, used by R's diptest package)
    n_extreme = 0
    for _ in range(n_boot):
        x_boot = rng.uniform(0, 1, n)
        dip_boot = _compute_dip_statistic(x_boot)
        if dip_boot >= dip_obs:
            n_extreme += 1

    p_value = (n_extreme + 1) / (n_boot + 1)

    return DipTestResult(
        dip=dip_obs,
        p_value=p_value,
        n_boot=n_boot,
        n_extreme=n_extreme,
    )


# ---------------------------------------------------------------------------
# Bimodality strength assessment
# ---------------------------------------------------------------------------


@dataclass
class BimodalityStrength:
    """Comprehensive bimodality strength assessment.

    Combines dip_ratio, h_crit ratio, and mode count into an
    interpretable strength score.

    Parameters
    ----------
    dip_ratio : float
        Valley-to-minimum-peak ratio (0 = strongly bimodal, 1 = unimodal).
    h_crit_ratio : float
        h_crit / silverman_bandwidth, higher = stronger bimodality.
    n_modes : int
        Number of KDE modes at the analysis bandwidth.
    strength : str
        Interpretive label: "strong", "moderate", "weak", or "unimodal".
    strength_score : float
        Scalar strength from 0.0 (unimodal) to 1.0 (strongly bimodal).
    is_heuristic : bool
        Always True for the current implementation.
    calibration_status : str
        Text tag describing the current calibration state.
    """

    dip_ratio: float
    h_crit_ratio: float
    n_modes: int
    strength: str
    strength_score: float
    is_heuristic: bool = True
    calibration_status: str = "heuristic"


def bimodality_strength(
    x: np.ndarray,
    h_factor: float = 0.85,
    kernel: Union[str, Callable] = "gaussian",
) -> BimodalityStrength:
    """Assess bimodality strength of a distribution.

    Computes a comprehensive assessment combining dip_ratio, the ratio of
    critical bandwidth to Silverman bandwidth, and the number of KDE modes
    at the analysis bandwidth (h = h_crit * h_factor).

    Parameters
    ----------
    x : np.ndarray
        1D array of input data.
    h_factor : float, optional
        Multiplier on critical bandwidth for analysis (default 0.85).
        Lower values produce deeper troughs (stronger bimodal signal),
        higher values approach the transition boundary.
    kernel : str or callable, optional
        Kernel function (default "gaussian").

    Returns
    -------
    BimodalityStrength
        Structured assessment with dip_ratio, h_crit_ratio, n_modes,
        and an interpretable strength label + score. The label is heuristic
        rather than a validated decision rule.

    Notes
    -----
    The strength_score is computed as:
    - If dip_ratio >= 1.0 or n_modes < 2: 0.0 (unimodal)
    - Otherwise: max(0, min(1, (1 - dip_ratio) * 0.6
        + min(h_crit_ratio / 5, 1) * 0.4))

    The strength label follows the dip_ratio thresholds:
    - dip_ratio < 0.30: "strong"
    - dip_ratio < 0.70: "moderate"
    - dip_ratio < 0.95: "weak"
    - dip_ratio >= 0.95 or n_modes < 2: "unimodal"
    """
    _validate_input(x)

    # 1. Compute critical bandwidth
    h_crit, ok = critical_bandwidth(x, kernel=kernel)
    if not ok:
        import warnings

        warnings.warn(
            "Critical bandwidth search did not fully converge; proceeding with best estimate."
        )

    # 2. Compute analysis bandwidth and Silverman bandwidth
    h_analysis = h_crit * h_factor
    h_silver = silverman_bandwidth(x)
    h_crit_ratio = h_crit / h_silver if h_silver > 0 else float("nan")

    # 3. Find modes at analysis bandwidth
    mode_result = find_modes(x, h_analysis, kernel=kernel)
    n_modes = mode_result.n_modes

    # 4. Compute dip_ratio via _trough_ratio (if bimodal)
    if n_modes >= 2:
        dip_ratio = _trough_ratio(x, h_analysis, kernel=kernel)
    else:
        dip_ratio = 1.0

    # 5. Compute strength_score
    if dip_ratio >= 1.0 or n_modes < 2:
        strength_score = 0.0
    else:
        dip_component = (1.0 - dip_ratio) * 0.6
        ratio_component = min(h_crit_ratio / 5.0, 1.0) * 0.4
        strength_score = min(1.0, max(0.0, dip_component + ratio_component))

    # 6. Determine strength label
    if dip_ratio < 0.30:
        strength = "strong"
    elif dip_ratio < 0.70:
        strength = "moderate"
    elif dip_ratio < 0.95:
        strength = "weak"
    else:
        strength = "unimodal"

    return BimodalityStrength(
        dip_ratio=max(0.0, min(1.0, dip_ratio)),
        h_crit_ratio=h_crit_ratio,
        n_modes=n_modes,
        strength=strength,
        strength_score=strength_score,
        is_heuristic=True,
        calibration_status="heuristic",
    )


# ---------------------------------------------------------------------------
# Excess mass test for multimodality
# ---------------------------------------------------------------------------


@dataclass
class ExcessMassResult:
    """Result of the excess mass test for multimodality.

    Parameters
    ----------
    n_modes_estimated : int
        Estimated number of modes (k with largest significant Δ_k).
    test_statistics : np.ndarray
        Δ_k = D_k - D_{k+1} for k = 0, ..., n_modes_max-2.
    p_values : np.ndarray
        Bootstrap p-values for each Δ_k.
    d_values : np.ndarray
        D_k values for k = 1, ..., n_modes_max.
    excess_mass_values : np.ndarray
        EM(λ) at each λ threshold.
    lambda_grid : np.ndarray
        λ values used.
    n_boot : int
        Number of bootstrap resamples.
    bandwidth : float
        Bandwidth used for KDE.
    """

    n_modes_estimated: int
    test_statistics: np.ndarray
    p_values: np.ndarray
    d_values: np.ndarray
    excess_mass_values: np.ndarray
    lambda_grid: np.ndarray
    n_boot: int
    bandwidth: float


def _excess_mass_compute(kde_vals: np.ndarray, dx: float, lam: float) -> Tuple[float, int]:
    """Compute excess mass at threshold λ.

    Parameters
    ----------
    kde_vals : np.ndarray
        KDE density values on a uniform grid.
    dx : float
        Grid spacing.
    lam : float
        Threshold λ ≥ 0.

    Returns
    -------
    em : float
        Excess mass = ∫ max(0, f(x) - λ) dx (trapezoidal rule).
    n_components : int
        Number of connected components where f(x) ≥ λ.
    """
    # Excess mass via trapezoidal integration
    above = np.maximum(kde_vals - lam, 0.0)
    em = np.trapezoid(above, dx=dx)

    # Count connected components at level λ
    binary = (kde_vals >= lam).astype(int)
    transitions = np.diff(binary, prepend=0, append=0)
    n_components = int(np.sum(transitions == 1))

    return float(em), n_components


def excess_mass(
    x: np.ndarray,
    h: Optional[float] = None,
    grid_points: int = 1000,
    kernel: Union[str, Callable] = "gaussian",
    n_lambda: int = 100,
    n_modes_max: int = 5,
    n_boot: int = 199,
    random_state: Optional[int] = None,
) -> ExcessMassResult:
    """Excess mass test for multimodality.

    Implements the excess mass test (Müller & Sawitzki, 1991) for detecting
    the number of modes in a distribution. The test measures how much
    probability mass "exceeds" what would be expected under a given number
    of modes.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data.
    h : float, optional
        Bandwidth for KDE. If None (default), uses Silverman's rule.
    grid_points : int, optional
        Minimum number of KDE evaluation points (default 1000).
    kernel : str or callable, optional
        Kernel function (default "gaussian").
    n_lambda : int, optional
        Number of λ threshold values (default 100).
    n_modes_max : int, optional
        Maximum number of modes to test (default 5).
    n_boot : int, optional
        Number of bootstrap resamples (default 199).
    random_state : int or None, optional
        Random seed for reproducible bootstrap.

    Returns
    -------
    ExcessMassResult
        Structured result with estimated mode count, test statistics, p-values.

    Notes
    -----
    The test statistic for k vs k+1 modes:
        D_k = max_λ [EM(λ) - λ·(k-1)]
        Δ_k = D_k - D_{k+1}

    Large positive Δ_k indicates data has at least k+1 modes.
    p-values are calibrated by sampling from Uniform(0,1) under the
    null hypothesis of at most k modes (least favorable unimodal distribution).

    References
    ----------
    Müller & Sawitzki (1991), "Excess Mass Estimates and Tests for
    Multimodality", JASA 86(415).
    """
    _validate_input(x)

    if h is None:
        h = silverman_bandwidth(x)
    if h <= 0:
        h = 0.01

    n = len(x)
    rng = np.random.default_rng(random_state)

    # --- KDE on a grid ---
    data_range = x.max() - x.min() if len(x) > 1 else 1.0
    gp = _kde_grid_points(len(x), h=h, data_range=data_range)
    gp = max(gp, grid_points)
    grid = np.linspace(x.min() - 3 * h, x.max() + 3 * h, gp)
    kde_vals = gaussian_kde(x, grid, h, kernel=kernel)
    dx = grid[1] - grid[0]

    # --- Lambda grid from 0 to max(KDE) ---
    lam_grid = np.linspace(0, float(np.max(kde_vals)), n_lambda)

    # --- Compute excess mass and component counts at each λ ---
    em_values = np.zeros(n_lambda)
    n_components = np.zeros(n_lambda, dtype=int)

    for i, lam in enumerate(lam_grid):
        em_values[i], n_components[i] = _excess_mass_compute(kde_vals, dx, lam)

    # --- Compute D_k for k = 1, ..., n_modes_max ---
    # D_k = max over λ where n_components >= k of [EM(λ) - λ·(k-1)]
    d_values = np.zeros(n_modes_max)
    for k in range(1, n_modes_max + 1):
        # Only consider λ where at least k components exist
        valid = n_components >= k
        if not np.any(valid):
            d_values[k - 1] = 0.0
        else:
            obj = em_values[valid] - lam_grid[valid] * (k - 1)
            d_values[k - 1] = float(np.max(obj))

    # --- Test statistics Δ_k = D_k - D_{k+1} ---
    n_deltas = n_modes_max - 1
    delta_obs = np.array([d_values[k] - d_values[k + 1] for k in range(n_deltas)])

    # --- Bootstrap calibration ---
    # Under H0 of at most k modes, sample from Uniform(0,1)
    extreme_counts = np.zeros(n_deltas)

    for _ in range(n_boot):
        x_boot = rng.uniform(0, 1, n)

        # KDE for bootstrap sample
        grid_boot = np.linspace(x_boot.min() - 3 * h, x_boot.max() + 3 * h, gp)
        kde_boot = gaussian_kde(x_boot, grid_boot, h, kernel=kernel)
        dx_boot = grid_boot[1] - grid_boot[0]

        # Lambda grid for bootstrap
        lam_boot = np.linspace(0, float(np.max(kde_boot)), n_lambda)

        em_boot = np.zeros(n_lambda)
        n_components_boot = np.zeros(n_lambda, dtype=int)
        for i, lam in enumerate(lam_boot):
            em_boot[i], n_components_boot[i] = _excess_mass_compute(kde_boot, dx_boot, lam)

        # D_k for bootstrap
        d_boot = np.zeros(n_modes_max)
        for k in range(1, n_modes_max + 1):
            valid = n_components_boot >= k
            if not np.any(valid):
                d_boot[k - 1] = 0.0
            else:
                obj = em_boot[valid] - lam_boot[valid] * (k - 1)
                d_boot[k - 1] = float(np.max(obj))

        delta_boot = np.array([d_boot[k] - d_boot[k + 1] for k in range(n_deltas)])

        for k in range(n_deltas):
            if delta_boot[k] >= delta_obs[k]:
                extreme_counts[k] += 1

    # --- p-values ---
    p_values = (extreme_counts + 1) / (n_boot + 1)

    # --- Estimate mode count ---
    # Δ_0 tests 1 vs 2+ modes, Δ_1 tests 2 vs 3+ modes, etc.
    # If p_value[k] < 0.05, we reject H0 of at most k+1 modes
    n_modes_est = 1
    for k in range(n_deltas):
        if p_values[k] < 0.05:
            n_modes_est = k + 2

    return ExcessMassResult(
        n_modes_estimated=n_modes_est,
        test_statistics=delta_obs,
        p_values=p_values,
        d_values=d_values,
        excess_mass_values=em_values,
        lambda_grid=lam_grid,
        n_boot=n_boot,
        bandwidth=h,
    )
