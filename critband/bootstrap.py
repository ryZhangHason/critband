"""
Bootstrap confidence interval estimation for critical bandwidth.

Provides resampling-based inference for the critical bandwidth parameter,
returning confidence intervals and standard errors.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf
from scipy.stats import bootstrap as scipy_bootstrap
from scipy.stats import beta as scipy_beta
from scipy.stats import t as scipy_t

from critband.bandwidth import (
    _compute_dip_statistic,
    _validate_input,
    critical_bandwidth,
    dip_test,
    excess_mass,
)


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

    if np.isfinite(ci_lower) and np.isfinite(ci_upper):
        ci_lower = float(min(ci_lower, h_crit_original))
        ci_upper = float(max(ci_upper, h_crit_original))

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
    calibration_method : str
        Calibration path used for the reported p-value.
    calibration_alpha : float
        Nominal alpha used by calibration-specific adjustments.
    calibration_factor : float
        Scale factor applied to the observed statistic before comparison.
    """

    h_crit: float
    p_value: float
    n_resamples: int
    n_extreme: int
    n_failed: int
    null_distribution: np.ndarray
    calibration_method: str = "silverman"
    calibration_alpha: float = 0.05
    calibration_factor: float = 1.0


@dataclass
class ModeTestResult:
    """R-like result object for multimodality tests.

    Attributes
    ----------
    statistic : float
        Test statistic reported by the selected method.
    p_value : float
        P-value for the selected test.
    null_value : int
        Null hypothesis value for the maximum number of modes.
    alternative : str
        Alternative hypothesis label, usually "greater".
    method : str
        Human-readable method description.
    sample_size : int
        Number of non-missing observations used in the test.
    data_name : str
        Name of the data source or array.
    bad_obs : int
        Number of missing or non-finite values removed before testing.
    statistic_name : str
        Name of the statistic, e.g. "Critical bandwidth".
    calibration_method : str
        Calibration path used internally, when relevant.
    """

    statistic: float
    p_value: float
    null_value: int
    alternative: str
    method: str
    sample_size: int
    data_name: str = "data"
    bad_obs: int = 0
    statistic_name: str = "Statistic"
    calibration_method: str = "n/a"


def _hall_york_lambda(alpha: float) -> float:
    """Return the Hall-York calibration factor for Silverman's test."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 for Hall-York calibration")
    numerator = (
        0.94029 * alpha**3
        - 1.59914 * alpha**2
        + 0.17695 * alpha
        + 0.48971
    )
    denominator = (
        alpha**3
        - 1.77793 * alpha**2
        + 0.36162 * alpha
        + 0.42423
    )
    return float(numerator / denominator)


def silverman_test(
    x: np.ndarray,
    n_resamples: int = 999,
    random_state: Optional[int] = None,
    mod0: int = 1,
    calibration: str = "silverman",
    alpha: float = 0.05,
    lowsup: Optional[float] = None,
    uppsup: Optional[float] = None,
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
    mod0 : int, optional
        Maximum number of modes in the null hypothesis (default 1).
    calibration : str, optional
        Calibration path for the p-value. Supported values are
        "silverman" (default) and "hall_york".
    alpha : float, optional
        Nominal alpha used by Hall-York calibration (default 0.05).
    lowsup : float, optional
        Lower support bound for bounded-support calibration.
    uppsup : float, optional
        Upper support bound for bounded-support calibration.
    **kwargs
        Passed through to critical_bandwidth().

    Returns
    -------
    SilvermanTestResult
        Result with h_crit, p-value, null distribution, and calibration metadata.

    Notes
    -----
    The test generates null samples using Silverman's method:
    1. Compute h_crit on original data
    2. Generate bootstrap samples from a KDE with bandwidth h_crit
       (the critical KDE — just barely unimodal)
    3. Apply variance correction to match original data variance
    4. p-value = n_extreme / n_resamples for the default Silverman path.
       Hall-York calibration rescales the observed statistic by lambda(alpha)
       before comparing it to the same null distribution.

    References
    ----------
    Silverman, B.W. (1981). Using Kernel Density Estimates to Investigate
    Multimodality. JRSS-B, 43(1), 97-99.
    """
    _validate_input(x)

    calibration_key = calibration.strip().lower().replace("-", "_").replace(" ", "_")
    if calibration_key in {"silverman", "si", "cbws"}:
        calibration_method = "silverman"
        calibration_factor = 1.0
    elif calibration_key in {"hall_york", "hallyork", "hy"}:
        calibration_method = "hall_york"
        calibration_factor = _hall_york_lambda(alpha)
    else:
        raise ValueError(
            "Unknown calibration method. Use 'silverman' or 'hall_york'."
        )

    # Compute observed test statistic
    k = mod0 + 1
    h_crit_obs, ok = critical_bandwidth(x, k=k, lowsup=lowsup, uppsup=uppsup, **kwargs)
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
        x_boot = _sample_bounded_kde_bootstrap(x, h_crit_obs, rng, lowsup, uppsup)

        # Step 2: Variance correction (scale back to original variance)
        var_boot = float(np.var(x_boot, ddof=1))
        if var_boot > 0 and x_var > 0:
            x_boot = x_mean + (x_boot - np.mean(x_boot)) * np.sqrt(x_var / var_boot)
            has_bounds = (
                (lowsup is not None and np.isfinite(lowsup))
                or (uppsup is not None and np.isfinite(uppsup))
            )
            if has_bounds:
                x_boot = np.clip(
                    x_boot,
                    -np.inf if lowsup is None else lowsup,
                    np.inf if uppsup is None else uppsup,
                )

        # Step 3: Compute h_crit on null sample
        h_null, ok_null = critical_bandwidth(
            x_boot, k=k, lowsup=lowsup, uppsup=uppsup, **kwargs
        )
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

    effective_obs = h_crit_obs * calibration_factor
    n_extreme = int(np.sum(null_arr > effective_obs))
    p_value = n_extreme / n_resamples if n_resamples > 0 else float("nan")

    return SilvermanTestResult(
        h_crit=h_crit_obs,
        p_value=p_value,
        n_resamples=n_resamples,
        n_extreme=n_extreme,
        n_failed=n_failed,
        null_distribution=null_arr,
        calibration_method=calibration_method,
        calibration_alpha=alpha,
        calibration_factor=calibration_factor,
    )


def _remove_nonfinite(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove non-finite observations and return cleaned data plus count."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Input must be 1-dimensional, got {arr.ndim} dimensions")
    bad_obs = int(np.sum(~np.isfinite(arr)))
    cleaned = arr[np.isfinite(arr)]
    if len(cleaned) == 0:
        raise ValueError("No observations remain after removing non-finite values")
    return cleaned, bad_obs


def _cramvm_statistic(data: np.ndarray, bw: float) -> float:
    """Compute the Fisher-Marron Cramér-von Mises statistic."""
    x = np.asarray(data, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan")
    if bw <= 0:
        return float("nan")

    diffs = (x[:, None] - x[None, :]) / bw
    fg = np.mean(0.5 * (1.0 + erf(diffs / np.sqrt(2.0))), axis=1)
    u = np.sort(fg)
    z = np.arange(1, n + 1, dtype=float)
    sumand = (u - (2.0 * z - 1.0) / (2.0 * n)) ** 2 + 1.0 / (12.0 * n)
    return float(np.sum(sumand))


def _sample_bounded_kde_bootstrap(
    x: np.ndarray,
    bandwidth: float,
    rng: np.random.Generator,
    lowsup: Optional[float],
    uppsup: Optional[float],
) -> np.ndarray:
    """Resample from a KDE with optional finite-support rejection."""
    n = len(x)
    sample = x[rng.integers(0, n, size=n)].astype(float)
    sample += bandwidth * rng.normal(0.0, 1.0, n)
    has_bounds = (
        (lowsup is not None and np.isfinite(lowsup))
        or (uppsup is not None and np.isfinite(uppsup))
    )
    if has_bounds:
        low = -np.inf if lowsup is None else lowsup
        high = np.inf if uppsup is None else uppsup
        for idx in range(n):
            tries = 0
            while (sample[idx] < low or sample[idx] > high) and tries < 100:
                sample[idx] = x[rng.integers(0, n)] + bandwidth * rng.normal()
                tries += 1
            if sample[idx] < low:
                sample[idx] = low
            elif sample[idx] > high:
                sample[idx] = high
    return sample


def _excess_mass_scalar(
    data: np.ndarray,
    mod0: int = 1,
    approximate: bool = False,
    gridsize: Optional[tuple[int, int]] = None,
) -> float:
    """Return a scalar excess-mass statistic compatible with multimode."""
    from critband.bandwidth import excess_mass as _excess_mass

    if mod0 == 1 and not approximate:
        return float(2.0 * _compute_dip_statistic(data))
    n_lambda = gridsize[1] if gridsize is not None else 100
    grid_points = gridsize[0] if gridsize is not None else 1000
    result = _excess_mass(
        data,
        n_lambda=n_lambda,
        n_modes_max=max(2, mod0 + 1),
        grid_points=grid_points,
    )
    idx = min(max(mod0 - 1, 0), len(result.d_values) - 1)
    return float(result.d_values[idx])


def _hall_york_method2_pvalue(
    x: np.ndarray,
    cbw_obs: float,
    B: int,
    nMC: int,
    BMC: int,
    lowsup: float,
    uppsup: float,
    n: int,
    tol: float,
    rng: np.random.Generator,
) -> float:
    """Approximate Hall-York method 2 calibration."""
    base = silverman_test(
        x,
        n_resamples=B,
        random_state=int(rng.integers(0, 2**32 - 1)),
        mod0=1,
        calibration="hall_york",
        alpha=0.05,
        lowsup=lowsup,
        uppsup=uppsup,
    )
    pv_sil = base.p_value
    pv_mc = []
    for _ in range(nMC):
        data_mc = rng.normal(size=len(x))
        cbw_mc, _ = critical_bandwidth(
            data_mc,
            k=1,
            lowsup=-1.5 if not np.isfinite(lowsup) else lowsup,
            uppsup=1.5 if not np.isfinite(uppsup) else uppsup,
            method="auto",
            tol=tol,
            ci_resamples=1,
        )
        cbw_bmc = []
        for _ in range(BMC):
            eps = rng.normal(0.0, cbw_mc, len(data_mc))
            samp = rng.choice(data_mc, size=len(data_mc), replace=True)
            data_bmc = samp + eps
            cbw_rep, _ = critical_bandwidth(
                data_bmc,
                k=1,
                lowsup=-1.5 if not np.isfinite(lowsup) else lowsup,
                uppsup=1.5 if not np.isfinite(uppsup) else uppsup,
                method="auto",
                tol=tol,
                ci_resamples=1,
            )
            cbw_bmc.append(cbw_rep)
        pv_mc.append(float(np.mean(cbw_mc < np.asarray(cbw_bmc))))
    return float(np.mean(np.asarray(pv_mc) < pv_sil))


def modetest(
    data: np.ndarray,
    mod0: int = 1,
    method: str = "ACR",
    B: int = 500,
    lowsup: float = -np.inf,
    uppsup: float = np.inf,
    submethod: Optional[int] = None,
    n: Optional[int] = None,
    tol: Optional[float] = None,
    tol2: Optional[float] = None,
    gridsize: Optional[Any] = None,
    alpha: Optional[float] = None,
    nMC: Optional[int] = None,
    BMC: Optional[int] = None,
    random_state: Optional[int] = None,
) -> ModeTestResult:
    """R-like multimodality test front door.

    This is a compatibility wrapper around the existing Python primitives.
    It matches the multimode::modetest() argument surface as closely as the
    current implementation allows.
    """
    if not isinstance(mod0, (int, np.integer)) or mod0 <= 0:
        raise ValueError("Argument 'mod0' must be a positive integer number")
    if B <= 0:
        raise ValueError("Argument 'B' must be a positive integer number")

    method_key = str(method).strip().upper()
    x, bad_obs = _remove_nonfinite(data)
    sample_size = int(len(x))
    if sample_size == 0:
        raise ValueError("No observations remain after removing non-finite values")

    # Normalize defaults roughly in line with multimode, while keeping the
    # current primitives stable.
    if submethod is None:
        submethod = 1
        if method_key == "ACR" and mod0 > 1 and sample_size > 200:
            submethod = 2
    if alpha is None:
        alpha = 0.05
    if n is None:
        n = 2**10 if method_key in {"SI", "HY", "FM"} else 2**15
    if tol is None:
        tol = 1e-5
    if tol2 is None:
        tol2 = 1e-5
    if nMC is None:
        nMC = 100
    if BMC is None:
        BMC = 100

    # The existing primitives only support the unimodal null directly for the
    # dip and excess-mass branches. Keep the contract explicit for now.
    if method_key in {"HY", "HH", "CH"} and mod0 != 1:
        raise NotImplementedError(
            f"method '{method_key}' is currently implemented only for mod0=1"
        )

    if method_key == "SI":
        if submethod == 1:
            result = silverman_test(
                x,
                n_resamples=B,
                random_state=random_state,
                mod0=mod0,
                calibration="silverman",
                lowsup=lowsup,
                uppsup=uppsup,
            )
            statistic = result.h_crit
            p_value = result.p_value
            calibration_method = result.calibration_method
        else:
            result = silverman_test(
                x,
                n_resamples=B,
                random_state=random_state,
                mod0=mod0,
                calibration="silverman",
                lowsup=lowsup,
                uppsup=uppsup,
            )
            statistic = result.h_crit
            p_value = result.p_value
            calibration_method = "silverman_submethod2"
        method_label = "Silverman (1981) critical bandwidth test"
        statistic_name = "Critical bandwidth"
    elif method_key == "HY":
        if submethod == 1:
            if not (
                lowsup is not None
                and uppsup is not None
                and np.isfinite(lowsup)
                and np.isfinite(uppsup)
            ):
                raise NotImplementedError(
                    "Hall-York parity currently requires finite lowsup and uppsup"
                )
            result = silverman_test(
                x,
                n_resamples=B,
                random_state=random_state,
                mod0=1,
                calibration="hall_york",
                alpha=alpha,
                lowsup=lowsup,
                uppsup=uppsup,
            )
            statistic = result.h_crit
            p_value = result.p_value
            calibration_method = result.calibration_method
        else:
            if not (
                lowsup is not None
                and uppsup is not None
                and np.isfinite(lowsup)
                and np.isfinite(uppsup)
            ):
                raise NotImplementedError(
                    "Hall-York parity currently requires finite lowsup and uppsup"
                )
            statistic = critical_bandwidth(
                x,
                k=2,
                lowsup=lowsup,
                uppsup=uppsup,
                tol=tol,
                method="auto",
            )[0]
            rng = np.random.default_rng(random_state)
            p_value = _hall_york_method2_pvalue(
                x,
                statistic,
                B=B,
                nMC=nMC,
                BMC=BMC,
                lowsup=lowsup,
                uppsup=uppsup,
                n=n,
                tol=tol,
                rng=rng,
            )
            calibration_method = "hall_york_submethod2"
        method_label = "Hall and York (2001) critical bandwidth test"
        statistic_name = "Critical bandwidth"
    elif method_key == "FM":
        bw_crit = critical_bandwidth(
            x,
            k=mod0 + 1,
            tol=tol,
            method="auto",
            lowsup=lowsup,
            uppsup=uppsup,
        )[0]
        statistic = _cramvm_statistic(x, bw_crit)
        rng = np.random.default_rng(random_state)
        stat_boot = []
        for _ in range(B):
            data_b = _sample_bounded_kde_bootstrap(x, bw_crit, rng, lowsup, uppsup)
            bw_b = critical_bandwidth(
                data_b,
                k=mod0 + 1,
                tol=tol,
                method="auto",
                lowsup=lowsup,
                uppsup=uppsup,
            )[0]
            stat_boot.append(_cramvm_statistic(data_b, bw_b))
        stat_boot_arr = np.asarray(stat_boot, dtype=float)
        p_value = float(np.mean(statistic < stat_boot_arr))
        method_label = "Fisher and Marron (2001) Cramer-von Mises test"
        statistic_name = "Cramer-von Mises"
        calibration_method = "bootstrap_cvm"
    elif method_key == "HH":
        dip = dip_test(x, n_boot=B, random_state=random_state)
        statistic = dip.dip
        p_value = dip.p_value
        method_label = "Hartigan and Hartigan (1985) dip test"
        statistic_name = "Dip"
        calibration_method = "uniform_bootstrap"
    elif method_key == "CH":
        statistic = _excess_mass_scalar(x, mod0=mod0, approximate=False)
        rng = np.random.default_rng(random_state)
        ndata = len(x)
        hest = ((4.0 / (3.0 * ndata)) ** (1.0 / 5.0)) * np.std(x, ddof=1)
        fest = np.histogram(x, bins=max(32, min(512, n)))[0]
        fmodest = float(np.max(fest) / max(1.0, ndata))
        hest2 = 0.94 * np.std(x, ddof=1) * (ndata ** (-1.0 / 9.0))
        dataest = (x[np.argmax(np.histogram(x, bins=max(32, min(512, n)))[0])] - x) / max(hest2, 1e-12)
        fdmodest = float(np.mean(((dataest) ** 2 - 1.0) * np.exp(-(dataest**2) / 2.0) / np.sqrt(2.0 * np.pi) / max(hest2**3, 1e-12)))
        d = abs(fdmodest) / max(fmodest**3, 1e-12)
        emB = []
        if d < 2 * np.pi:
            def _fun1(xx: float) -> float:
                return (scipy_beta.cdf(xx, xx, xx)) ** 2 * 2 ** (4 * xx - 1) * (xx - 1) - d
            try:
                betaest = brentq(_fun1, 1.0, 256.25)
            except ValueError:
                betaest = 1.0
            for _ in range(B):
                nuevdat = scipy_beta.rvs(betaest, betaest, size=ndata, random_state=rng)
                emB.append(_excess_mass_scalar(nuevdat, mod0=mod0, approximate=False))
        else:
            def _fun2(xx: float) -> float:
                return 2 * (scipy_beta.cdf(xx - 0.5, 0.5, 0.5)) ** 2 * xx - d
            try:
                betaest = brentq(_fun2, 0.5, 2**8)
            except ValueError:
                betaest = 2.0
            for _ in range(B):
                nuevdat = scipy_t.rvs(2 * betaest - 1, size=ndata, random_state=rng)
                nuevdat = nuevdat / np.sqrt(max(2 * betaest - 1, 1e-12))
                emB.append(_excess_mass_scalar(nuevdat, mod0=mod0, approximate=False))
        p_value = float(np.mean(statistic < np.asarray(emB)))
        method_label = "Cheng and Hall (1998) excess mass test"
        statistic_name = "Excess mass"
        calibration_method = "excess_mass_bootstrap"
    elif method_key == "ACR":
        statistic = _excess_mass_scalar(
            x,
            mod0=mod0,
            approximate=(submethod == 2),
            gridsize=(int(gridsize[0]), int(gridsize[1])) if gridsize is not None else None,
        )
        em = excess_mass(
            x,
            n_boot=B,
            n_modes_max=max(2, mod0 + 1),
            random_state=random_state,
            lowsup=lowsup,
            uppsup=uppsup,
        )
        p_value = float(em.p_values[min(mod0 - 1, len(em.p_values) - 1)])
        method_label = "Ameijeiras-Alonso et al. (2019) excess mass test"
        statistic_name = "Excess mass"
        calibration_method = "excess_mass_bootstrap" if submethod == 1 else "excess_mass_approx"
    else:
        raise ValueError(
            "Unknown method. Use one of SI, HY, FM, HH, CH, ACR."
        )

    return ModeTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        null_value=int(mod0),
        alternative="greater",
        method=method_label,
        sample_size=sample_size,
        data_name="data",
        bad_obs=bad_obs,
        statistic_name=statistic_name,
        calibration_method=calibration_method,
    )
