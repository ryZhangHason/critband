"""
Bootstrap confidence interval estimation for critical bandwidth.

Provides resampling-based inference for the critical bandwidth parameter,
returning confidence intervals and standard errors.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

from critband.bandwidth import _validate_input, critical_bandwidth


@dataclass
class BootstrapResult:
    """
    Result of a bootstrap critical bandwidth analysis.

    Attributes
    ----------
    h_crit : float
        Critical bandwidth on the original (non-resampled) data.
    ci_lower : float
        Lower bound of the (1 - alpha) percentile confidence interval.
    ci_upper : float
        Upper bound of the (1 - alpha) percentile confidence interval.
    standard_error : float
        Bootstrap standard error (std of resampled h_crit distribution).
    distribution : np.ndarray
        Array of successful bootstrap h_crit values (failed samples excluded).
    n_resamples : int
        Number of bootstrap resamples attempted.
    confidence_level : float
        The confidence level used (1 - alpha).
    n_failed : int
        Number of bootstrap resamples that failed to converge.
    interval_method : str
        Bootstrap interval method used to compute ci_lower/ci_upper.
    """

    h_crit: float
    ci_lower: float
    ci_upper: float
    standard_error: float
    distribution: np.ndarray
    n_resamples: int
    confidence_level: float
    n_failed: int = 0
    interval_method: str = "percentile"


def bootstrap_critical_bandwidth(
    x: np.ndarray,
    n_resamples: int = 999,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    ci_method: str = "percentile",
    **kwargs: Any,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for the critical bandwidth.

    Resamples the input data with replacement, computes h_crit for each
    resample, and returns a bootstrap interval and standard error.

    Parameters
    ----------
    x : np.ndarray
        1D array of input data.
    n_resamples : int, optional
        Number of bootstrap resamples (default 999).
    alpha : float, optional
        Significance level for the confidence interval (default 0.05 -> 95% interval).
    random_state : int, optional
        Random seed for reproducible resampling.
    ci_method : str, optional
        Bootstrap interval method. Supported values are "BCa", "percentile",
        and "basic". Default is "percentile".
    **kwargs
        Additional keyword arguments passed through to critical_bandwidth().
        See critical_bandwidth() documentation for supported options
        (h_min, h_max, tol, max_iter, method, kernel).

    Returns
    -------
    BootstrapResult
        Structured result with h_crit, CI bounds, standard error, and
        the full bootstrap distribution.

    Notes
    -----
    The default interval method uses SciPy's percentile bootstrap routine.
    If SciPy's interval computation fails or returns a degenerate interval,
    the function falls back to a manual percentile bootstrap so that a
    usable exploratory interval is still returned.

    Bootstrap resamples where critical_bandwidth fails to converge
    (success=False) are currently kept in the distribution as the best
    estimate returned by critical_bandwidth().
    """
    _validate_input(x)

    rng = np.random.default_rng(random_state)

    # Critical bandwidth on original data
    h_crit_original, _ = critical_bandwidth(x, **kwargs)

    n = len(x)
    n_failed = 0

    def _statistic(sample: np.ndarray) -> float:
        h_crit_boot, _ = critical_bandwidth(sample, **kwargs)
        return float(h_crit_boot)

    def _manual_bootstrap() -> tuple[np.ndarray, int]:
        boot_samples: list[float] = []
        n_failed_local = 0
        for _ in range(n_resamples):
            resample = rng.choice(x, size=n, replace=True)
            h_crit_boot, ok = critical_bandwidth(resample, **kwargs)
            boot_samples.append(float(h_crit_boot))
            if not ok:
                n_failed_local += 1
        return np.asarray(boot_samples, dtype=float), n_failed_local

    def _all_failed_result(interval_method: str, boot_arr: np.ndarray, n_failed_local: int) -> BootstrapResult:
        return BootstrapResult(
            h_crit=h_crit_original,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            standard_error=float("nan"),
            distribution=boot_arr,
            n_resamples=n_resamples,
            confidence_level=1 - alpha,
            n_failed=n_failed_local,
            interval_method=interval_method,
        )

    scipy_method = ci_method.strip()
    scipy_method_map = {
        "bca": "BCa",
        "bca ": "BCa",
        "percentile": "percentile",
        "basic": "basic",
    }
    scipy_method_key = scipy_method_map.get(scipy_method.lower(), scipy_method)

    if scipy_method_key == "percentile":
        boot_arr, n_failed = _manual_bootstrap()
        if len(boot_arr) == 0:
            return _all_failed_result("percentile", boot_arr, n_failed)
        if n_failed == n_resamples:
            return _all_failed_result("percentile", boot_arr, n_failed)
        p_low = 100 * alpha / 2
        p_high = 100 * (1 - alpha / 2)
        ci_lower, ci_upper = np.percentile(boot_arr, [p_low, p_high])
        interval_method = "percentile"
    elif scipy_method_key == "basic":
        boot_arr, n_failed = _manual_bootstrap()
        if len(boot_arr) == 0:
            return _all_failed_result("basic", boot_arr, n_failed)
        if n_failed == n_resamples:
            return _all_failed_result("basic", boot_arr, n_failed)
        p_low = 100 * alpha / 2
        p_high = 100 * (1 - alpha / 2)
        q_low, q_high = np.percentile(boot_arr, [p_low, p_high])
        ci_lower = float(2 * h_crit_original - q_high)
        ci_upper = float(2 * h_crit_original - q_low)
        interval_method = "basic"
    else:
        scipy_error: Exception | None = None
        use_manual_percentile = False
        import warnings

        try:
            scipy_result = scipy_bootstrap(
                (x,),
                _statistic,
                n_resamples=n_resamples,
                confidence_level=1 - alpha,
                method=scipy_method_key,
                random_state=rng,
                vectorized=False,
                paired=False,
            )
            boot_arr = np.asarray(scipy_result.bootstrap_distribution, dtype=float).reshape(-1)
            ci_lower = float(scipy_result.confidence_interval.low)
            ci_upper = float(scipy_result.confidence_interval.high)
            interval_method = str(scipy_method_key)
            if (
                not np.isfinite(ci_lower)
                or not np.isfinite(ci_upper)
                or ci_lower >= ci_upper
            ):
                scipy_error = ValueError("degenerate confidence interval")
                use_manual_percentile = True
        except Exception as exc:
            scipy_error = exc
            use_manual_percentile = True

        if use_manual_percentile:
            warnings.warn(
                f"SciPy bootstrap failed ({scipy_error}); falling back to manual percentile resampling."
            )
            boot_arr, n_failed = _manual_bootstrap()
            if len(boot_arr) == 0:
                return _all_failed_result("percentile", boot_arr, n_failed)
            if n_failed == n_resamples:
                return _all_failed_result("percentile", boot_arr, n_failed)
            p_low = 100 * alpha / 2
            p_high = 100 * (1 - alpha / 2)
            ci_lower, ci_upper = np.percentile(boot_arr, [p_low, p_high])
            interval_method = "percentile"

    # Standard error: guard against single-element distribution
    if len(boot_arr) < 2:
        standard_error = float("nan")
    else:
        standard_error = float(np.std(boot_arr, ddof=1))

    return BootstrapResult(
        h_crit=h_crit_original,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        standard_error=standard_error,
        distribution=boot_arr,
        n_resamples=n_resamples,
        confidence_level=1 - alpha,
        n_failed=n_failed,
        interval_method=interval_method,
    )


@dataclass
class SilvermanTestResult:
    """Result of Silverman's test for bimodality.

    Attributes
    ----------
    h_crit : float
        Critical bandwidth on original data (test statistic).
    p_value : float
        Bootstrap p-value: proportion of null resamples with h_crit >= observed.
        Lower values indicate stronger evidence against unimodality.
    n_resamples : int
        Number of bootstrap resamples under H0.
    n_extreme : int
        Number of null resamples where h_crit >= observed h_crit.
    n_failed : int
        Number of null resamples that did not fully converge.
    null_distribution : np.ndarray
        Array of h_crit values from null resamples.
    """

    h_crit: float
    p_value: float
    n_resamples: int
    n_extreme: int
    n_failed: int
    null_distribution: np.ndarray


def silverman_test(
    x: np.ndarray,
    n_resamples: int = 999,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> SilvermanTestResult:
    """Silverman's (1981) bootstrap test for bimodality.

    Tests the null hypothesis that the underlying distribution has at most
    one mode (unimodal) against the alternative that it has two or more
    modes (bimodal/multimodal).

    Parameters
    ----------
    x : np.ndarray
        1D array of input data.
    n_resamples : int, optional
        Number of bootstrap resamples under H0 (default 999).
    random_state : int, optional
        Random seed for reproducible resampling.
    **kwargs
        Passed through to critical_bandwidth().

    Returns
    -------
    SilvermanTestResult
        Result with h_crit, p-value, null distribution.

    Notes
    -----
    The test generates null samples using Silverman's method:
    1. Compute h_crit on original data
    2. Generate bootstrap samples from a KDE with bandwidth h_crit
       (the critical KDE — just barely unimodal)
    3. Apply variance correction to match original data variance
    4. p-value = (n_extreme + 1) / (n_success + 1)

    References
    ----------
    Silverman, B.W. (1981). Using Kernel Density Estimates to Investigate
    Multimodality. JRSS-B, 43(1), 97-99.
    """
    _validate_input(x)

    # Compute observed test statistic
    h_crit_obs, ok = critical_bandwidth(x, **kwargs)
    if not ok:
        import warnings

        warnings.warn(
            "Critical bandwidth did not fully converge on original data; "
            "proceeding with best estimate."
        )

    n = len(x)
    rng = np.random.default_rng(random_state)
    x_mean = float(np.mean(x))
    x_var = float(np.var(x, ddof=1))

    null_h_crits: list[float] = []
    n_failed = 0

    for _ in range(n_resamples):
        # Step 1: Sample from KDE with bandwidth = h_crit_obs
        indices = rng.integers(0, n, size=n)
        x_boot = x[indices].astype(float)
        x_boot += h_crit_obs * rng.normal(0, 1, n)

        # Step 2: Variance correction (scale back to original variance)
        var_boot = float(np.var(x_boot, ddof=1))
        if var_boot > 0 and x_var > 0:
            x_boot = x_mean + (x_boot - np.mean(x_boot)) * np.sqrt(x_var / var_boot)

        # Step 3: Compute h_crit on null sample
        h_null, ok_null = critical_bandwidth(x_boot, **kwargs)
        null_h_crits.append(float(h_null))
        if not ok_null:
            n_failed += 1

    if n_failed > 0:
        import warnings

        warnings.warn(
            f"{n_failed}/{n_resamples} null resamples did not fully converge; "
            "best estimates were retained in the null distribution.",
            stacklevel=2,
        )

    null_arr = np.array(null_h_crits)

    # p-value with continuity correction (Davison & Hinkley)
    n_extreme = int(np.sum(null_arr >= h_crit_obs))
    p_value = (n_extreme + 1) / (n_resamples + 1)

    return SilvermanTestResult(
        h_crit=h_crit_obs,
        p_value=p_value,
        n_resamples=n_resamples,
        n_extreme=n_extreme,
        n_failed=n_failed,
        null_distribution=null_arr,
    )
