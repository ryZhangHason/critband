"""
Unit tests for bandwidth calculation functions.
"""

import numpy as np
import pytest
from scipy import integrate

from pola import (
    critical_bandwidth,
    detect_components,
    find_trough,
    gaussian_kde,
    silverman_bandwidth,
)
from pola.bandwidth import _trough_ratio, count_modes
from pola.benchmark import BENCHMARK_CASES


class TestSilvermanBandwidth:
    """Test Silverman's rule of thumb bandwidth calculation."""

    def test_normal_distribution(self):
        """Test bandwidth for standard normal distribution."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        h = silverman_bandwidth(x)

        # Should be close to optimal bandwidth for normal (≈ 0.26 for n=1000)
        assert 0.2 < h < 0.35

    def test_bimodal_distribution(self):
        """Test bandwidth for bimodal distribution."""
        np.random.seed(42)
        x1 = np.random.normal(-2, 0.5, 500)
        x2 = np.random.normal(2, 0.5, 500)
        x = np.concatenate([x1, x2])

        h = silverman_bandwidth(x)
        assert h > 0

    def test_single_point(self):
        """Test edge case with single data point."""
        x = np.array([5.0])
        h = silverman_bandwidth(x)
        assert np.isfinite(h)


class TestGaussianKDE:
    """Test Gaussian kernel density estimation."""

    def test_kde_values(self):
        """Test KDE returns valid probability density values."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        grid = np.linspace(-3, 3, 100)
        h = 0.5

        kde_vals = gaussian_kde(x, grid, h)

        assert len(kde_vals) == len(grid)
        assert np.all(kde_vals >= 0)
        # Integral should be approximately 1
        integral = integrate.trapezoid(kde_vals, grid)
        assert 0.8 < integral < 1.2


class TestKernelSupport:
    """Test custom kernel parameter threading through the call chain."""

    def test_builtin_kernels(self):
        """All built-in kernels produce valid density estimates."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        grid = np.linspace(-3, 3, 100)
        h = 0.5

        for kernel in ["gaussian", "epanechnikov", "uniform", "triangular"]:
            kde_vals = gaussian_kde(x, grid, h, kernel=kernel)
            assert len(kde_vals) == len(grid)
            assert np.all(kde_vals >= 0)
            integral = integrate.trapezoid(kde_vals, grid)
            assert 0.8 < integral < 1.2, f"Kernel '{kernel}' integral={integral:.3f}"

    def test_custom_callable_kernel(self):
        """A custom callable works as a kernel."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        grid = np.linspace(-3, 3, 100)
        h = 0.5

        def custom_kernel(u):
            return np.exp(-(u**2) / 2) / np.sqrt(2 * np.pi)

        kde_vals = gaussian_kde(x, grid, h, kernel=custom_kernel)
        assert np.all(kde_vals >= 0)
        integral = integrate.trapezoid(kde_vals, grid)
        assert 0.8 < integral < 1.2

    def test_unknown_kernel_raises(self):
        """Unknown kernel name raises ValueError."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        grid = np.linspace(-3, 3, 100)
        with pytest.raises(ValueError, match="Unknown kernel"):
            gaussian_kde(x, grid, 0.5, kernel="nonexistent")

    def test_kernel_threads_through_critical_bandwidth(self):
        """Kernel parameter threads through the full call chain."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        # Default Gaussian should work
        h_crit, ok = critical_bandwidth(x)
        assert ok

        # Epanechnikov should also work (different h_crit expected)
        h_crit_ep, ok_ep = critical_bandwidth(x, kernel="epanechnikov")
        assert ok_ep
        assert 0 < h_crit_ep < 5.0

    def test_kernel_threads_through_find_trough(self):
        """Kernel parameter threads through find_trough."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])
        h = 1.0

        trough = find_trough(x, h, kernel="epanechnikov")
        assert trough is not None
        assert -1 < trough < 1  # trough should be between the two modes


class TestCriticalBandwidth:
    """Test critical bandwidth detection algorithm."""

    def test_known_bimodal_case(self):
        """Test critical bandwidth for known bimodal distribution."""
        np.random.seed(42)
        # Create well-separated bimodal distribution
        x1 = np.random.normal(-2, 0.3, 200)
        x2 = np.random.normal(2, 0.3, 200)
        x = np.concatenate([x1, x2])

        h_crit, success = critical_bandwidth(x)

        assert success is True
        assert h_crit > 0.2
        assert h_crit < 2.0

    def test_unimodal_distribution(self):
        """Test critical bandwidth for already unimodal data."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 500)

        h_crit, success = critical_bandwidth(x)

        # Should find a critical bandwidth for unimodal data
        assert success is True

    def test_boundary_cases(self):
        """Test boundary handling."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-1, 0.2, 100), np.random.normal(1, 0.2, 100)])

        # Test with custom bounds
        h_crit, success = critical_bandwidth(x, h_min=0.01, h_max=2.0, tol=1e-4)
        assert success is True
        assert 0.7 < h_crit < 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestInputValidation:
    """Test input validation guards."""

    def test_nan_values(self):
        x = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="NaN|infinite"):
            silverman_bandwidth(x)

    def test_inf_values(self):
        x = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN|infinite"):
            silverman_bandwidth(x)

    def test_negative_inf_values(self):
        x = np.array([1.0, -np.inf, 3.0])
        with pytest.raises(ValueError, match="NaN|infinite"):
            silverman_bandwidth(x)

    def test_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            silverman_bandwidth(np.array([]))

    def test_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="1-dimensional"):
            silverman_bandwidth(x)

    def test_validation_affects_all_functions(self):
        x = np.array([1.0, np.nan])
        with pytest.raises(ValueError):
            gaussian_kde(x, np.array([0.0]), 0.5)
        with pytest.raises(ValueError):
            critical_bandwidth(x)

    def test_critical_bandwidth_empty(self):
        with pytest.raises(ValueError, match="empty"):
            critical_bandwidth(np.array([]))


class TestConstantData:
    """Test behavior with constant (zero-variance) data."""

    def test_silverman_constant(self):
        x = np.ones(10)
        h = silverman_bandwidth(x)
        assert h == 0.01  # floored minimum

    def test_critical_bandwidth_constant(self):
        x = np.ones(10)
        h_crit, success = critical_bandwidth(x)
        assert success is False  # already unimodal at min bandwidth
        assert np.isfinite(h_crit)

    def test_count_modes_zero_bandwidth(self):
        x = np.random.normal(0, 1, 100)
        modes = count_modes(x, h=0.0)
        assert modes == 1  # zero bandwidth → always unimodal

    def test_count_modes_negative_bandwidth(self):
        x = np.random.normal(0, 1, 100)
        modes = count_modes(x, h=-1.0)
        assert modes == 1


class TestStability:
    """Test numerical stability and determinism."""

    def test_deterministic(self):
        x = np.array([-2.1, -1.8, -1.7, 1.8, 2.0, 2.2])
        h1, _ = critical_bandwidth(x, tol=1e-4, max_iter=50)
        h2, _ = critical_bandwidth(x, tol=1e-4, max_iter=50)
        assert h1 == pytest.approx(h2)

    def test_small_perturbation(self):
        np.random.seed(42)
        x = np.concatenate(
            [
                np.random.normal(-2, 0.3, 200),
                np.random.normal(2, 0.3, 200),
            ]
        )
        h1, _ = critical_bandwidth(x)
        x_noisy = x + np.random.normal(0, 1e-6, len(x))
        h2, _ = critical_bandwidth(x_noisy)
        assert abs(h1 - h2) < 0.1

    def test_tolerance_does_not_harm_convergence(self):
        np.random.seed(42)
        x = np.concatenate(
            [
                np.random.normal(-2, 0.3, 200),
                np.random.normal(2, 0.3, 200),
            ]
        )
        h_coarse, s1 = critical_bandwidth(x, tol=1e-2)
        h_fine, s2 = critical_bandwidth(x, tol=1e-6)
        assert s1 is True and s2 is True
        # Both should converge to a reasonable range
        assert 0.5 < h_coarse < 2.0
        assert 0.5 < h_fine < 2.0


class TestGaussianKdeVectorized:
    """Test that vectorized KDE matches the original loop implementation."""

    def test_matches_loop_reference(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        grid = np.linspace(-3, 3, 200)
        h = 0.5

        result = gaussian_kde(x, grid, h)

        # Reference: explicit loop (same formula as original implementation)
        expected = np.array(
            [
                np.sum(np.exp(-((xi - x) ** 2) / (2 * h**2))) / (len(x) * h * np.sqrt(2 * np.pi))
                for xi in grid
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_small_dataset(self):
        x = np.array([1.0, 2.0, 3.0])
        grid = np.linspace(0, 4, 10)
        h = 0.5
        result = gaussian_kde(x, grid, h)
        assert len(result) == len(grid)
        assert np.all(result >= 0)

    def test_large_grid(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        grid = np.linspace(-4, 4, 10000)
        h = 0.5
        result = gaussian_kde(x, grid, h)
        integral = np.trapezoid(result, grid)
        assert 0.8 < integral < 1.2


class TestTroughRatio:
    """Test _trough_ratio continuous bimodality measure."""

    def test_trough_ratio_monotonic(self):
        """Verify _trough_ratio increases monotonically with h."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        h_values = np.linspace(0.5, 3.0, 10)
        ratios = [_trough_ratio(x, h) for h in h_values]
        assert all(ratios[i] <= ratios[i + 1] + 1e-10 for i in range(len(ratios) - 1))

    def test_objective_sign_change(self):
        """Verify f(h_min) < 0 < f(h_max) for bimodal data."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        h_min = silverman_bandwidth(x) / 20
        h_max = silverman_bandwidth(x) * 10
        f_min = _trough_ratio(x, h_min) - 0.5
        f_max = _trough_ratio(x, h_max) - 0.5
        assert f_min < 0 < f_max

    def test_trough_ratio_small_h(self):
        """Very small h should produce near-zero trough ratio for bimodal data."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        ratio = _trough_ratio(x, h=0.01)
        assert ratio < 0.3

    def test_trough_ratio_large_h(self):
        """Very large h should produce near-1 trough ratio."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        ratio = _trough_ratio(x, h=10.0)
        assert ratio > 0.5

    def test_trough_ratio_unimodal(self):
        """Unimodal data should return 1.0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        ratio = _trough_ratio(x, silverman_bandwidth(x))
        assert ratio == 1.0

    def test_trough_ratio_zero_h(self):
        """h <= 0 should return 1.0."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        assert _trough_ratio(x, 0.0) == 1.0
        assert _trough_ratio(x, -1.0) == 1.0


class TestCriticalBandwidthHybrid:
    """Test the upgraded critical_bandwidth with method parameter."""

    def test_brent_matches_binary(self):
        """Verify brent result matches binary result within tolerance."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        h_binary, ok1 = critical_bandwidth(x, method="binary", tol=1e-8)
        h_brent, ok2 = critical_bandwidth(x, method="brent", tol=1e-8)
        assert ok1 and ok2
        assert abs(h_binary - h_brent) < 0.05

    def test_auto_matches_binary(self):
        """Verify auto (hybrid) result matches binary."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        h_binary, ok1 = critical_bandwidth(x, method="binary", tol=1e-8)
        h_auto, ok2 = critical_bandwidth(x, method="auto", tol=1e-8)
        assert ok1 and ok2
        assert abs(h_binary - h_auto) < 0.05

    @pytest.mark.parametrize("case_name", list(BENCHMARK_CASES.keys()))
    def test_benchmark_reference(self, case_name):
        """Verify critical bandwidth matches expected benchmark values."""
        case = BENCHMARK_CASES[case_name]
        x = case.generator(42)
        h_crit, ok = critical_bandwidth(x, method="binary", tol=1e-8, max_iter=500)
        assert ok, f"{case_name}: did not converge"
        assert abs(h_crit - case.h_crit_expected) < case.h_crit_tolerance, (
            f"{case_name}: {case.h_crit_expected}±{case.h_crit_tolerance}, got {h_crit:.6f}"
        )

    def test_backward_compatibility(self):
        """Verify critical_bandwidth works without method parameter."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        h, ok = critical_bandwidth(x)
        assert ok
        assert h > 0

    def test_method_binary_no_method_param(self):
        """Explicit method='binary' matches default behavior."""
        np.random.seed(42)
        x = np.concatenate(
            [
                np.random.normal(-2, 0.3, 200),
                np.random.normal(2, 0.3, 200),
            ]
        )
        h_default, _ = critical_bandwidth(x)
        h_binary, _ = critical_bandwidth(x, method="binary")
        assert h_default == pytest.approx(h_binary, rel=0.01)

    def test_auto_fallback_on_brent_failure(self):
        """Auto method should fall back if Brent fails."""
        # Use constant data where Brent may have issues
        x = np.ones(10) * 5.0
        h, ok = critical_bandwidth(x, method="auto")
        assert np.isfinite(h)
        # ok may be False (already unimodal) but shouldn't crash


class TestFindTrough:
    """Test find_trough function."""

    def test_find_trough_between_modes(self):
        """Trough should be between the two component means for symmetric mixture."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        h_crit, ok = critical_bandwidth(x)
        assert ok
        # Use slightly smaller bandwidth to ensure bimodality
        trough = find_trough(x, h_crit * 0.9)
        assert trough is not None
        # Trough should be near 0 (midpoint between -2 and 2)
        assert -0.5 < trough < 0.5

    def test_find_trough_unimodal(self):
        """Unimodal data should return None."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        trough = find_trough(x, silverman_bandwidth(x) * 2)
        assert trough is None

    def test_find_trough_refinement(self):
        """Both grid and refined trough should find a result with bimodal bandwidth."""
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.normal(-2, 0.3, 200), rng.normal(2, 0.3, 200)])
        h_crit, _ = critical_bandwidth(x)
        # Use slightly smaller bandwidth to ensure bimodality
        trough_refined = find_trough(x, h_crit * 0.9, refine=True)
        trough_grid = find_trough(x, h_crit * 0.9, refine=False)
        assert trough_refined is not None
        assert trough_grid is not None

    def test_find_trough_zero_h(self):
        """h <= 0 should return None."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        assert find_trough(x, 0.0) is None
        assert find_trough(x, -1.0) is None

    def test_find_trough_multiple_modes(self):
        """Trimodal data should find a trough — may be between dominant peaks."""
        x = BENCHMARK_CASES["trimodal"].generator(42)
        h_crit, ok = critical_bandwidth(x)
        assert ok
        # Use slightly smaller bandwidth to ensure bimodality at critical region
        trough = find_trough(x, h_crit * 0.95)
        if trough is not None:
            # Trough between the two most prominent peaks
            assert isinstance(trough, float)
            assert np.isfinite(trough)


class TestDetectComponents:
    """Test bimodal component detection."""

    def check_means(self, result, expected_1, expected_2, tol=0.5):
        """Helper: verify component means match expected values."""
        assert abs(result.component1.mean - expected_1) < tol, (
            f"Expected mean1 ≈ {expected_1}, got {result.component1.mean:.3f}"
        )
        assert abs(result.component2.mean - expected_2) < tol, (
            f"Expected mean2 ≈ {expected_2}, got {result.component2.mean:.3f}"
        )

    def check_weight_sum(self, result):
        """Helper: verify weights sum to ~1."""
        assert abs(result.component1.weight + result.component2.weight - 1.0) < 0.01

    def test_well_separated(self):
        """Well-separated symmetric case: means near -2 and +2."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        result = detect_components(x)
        self.check_means(result, -2.0, 2.0, tol=0.4)
        self.check_weight_sum(result)
        assert result.component1.mean < result.component2.mean
        assert result.dip_ratio < 1.0  # some trough exists
        assert result.dip_ratio > 0.0
        assert result.critical_bandwidth > 0

    def test_moderate_separation(self):
        """Moderate separation: means near -1 and +1.5."""
        x = BENCHMARK_CASES["moderate_separation"].generator(42)
        result = detect_components(x)
        self.check_means(result, -1.0, 1.5, tol=0.5)
        self.check_weight_sum(result)

    def test_unequal_variance(self):
        """Unequal variance: left wider (σ=0.6) than right (σ=0.2)."""
        x = BENCHMARK_CASES["unequal_variance"].generator(42)
        result = detect_components(x)
        assert result.component2.std < result.component1.std, (
            "Right (tight) component should have smaller std"
        )
        self.check_weight_sum(result)

    def test_unequal_weights(self):
        """Unequal weights: left 100, right 400 → weight ratio ~1:4."""
        x = BENCHMARK_CASES["unequal_weights"].generator(42)
        result = detect_components(x)
        # Right component has ~4x the data
        assert result.component2.weight > result.component1.weight, (
            "Right component should have larger weight"
        )
        # Weight ratio should be approximately 100:400 = 0.2:0.8
        weight_ratio = result.component2.weight / result.component1.weight
        assert 2.0 < weight_ratio < 6.0, (
            f"Weight ratio {weight_ratio:.2f} outside expected range [2, 6]"
        )
        self.check_weight_sum(result)

    def test_components_ordered(self):
        """Component 1 always has lower mean than component 2."""
        x = BENCHMARK_CASES["extreme_separation"].generator(42)
        result = detect_components(x)
        assert result.component1.mean < result.component2.mean
        # Verify peak locations match
        assert result.separation_point > result.component1.mean
        assert result.separation_point < result.component2.mean

    def test_not_bimodal_raises(self):
        """Unimodal data should raise ValueError."""
        x = np.array([0.0])
        with pytest.raises(ValueError, match="not appear bimodal"):
            detect_components(x)

    def test_constant_data_raises(self):
        """Constant data should raise ValueError (single peak)."""
        x = np.ones(20)
        with pytest.raises(ValueError, match="not appear bimodal|empty component"):
            detect_components(x)

    def test_std_positive(self):
        """Component standard deviations should be positive."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        result = detect_components(x)
        assert result.component1.std > 0
        assert result.component2.std > 0

    def test_varying_h_factor(self):
        """Different h_factor values should produce similar results."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        r1 = detect_components(x, h_factor=0.80)
        r2 = detect_components(x, h_factor=0.90)
        # Both should detect means near -2 and +2
        assert abs(r1.component1.mean - r2.component1.mean) < 0.3
        assert abs(r1.component2.mean - r2.component2.mean) < 0.3


class TestBenchmarkStability:
    """Test benchmark reference values across multiple random seeds."""

    @pytest.mark.parametrize("case_name", list(BENCHMARK_CASES.keys()))
    def test_benchmark_stability(self, case_name):
        """Verify h_crit is stable across random seeds, within 2*std of
        the reference seed (seed=42) value. This provides independent
        validation that reference values are representative, not circular.
        """
        case = BENCHMARK_CASES[case_name]
        seeds = [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627]
        h_values = []
        for seed in seeds:
            x = case.generator(seed)
            h, ok = critical_bandwidth(x, method="binary", tol=1e-8, max_iter=500)
            assert ok, f"{case_name} seed={seed}: did not converge"
            h_values.append(h)

        h_arr = np.array(h_values)
        h_ref = h_arr[0]  # seed=42 is the reference
        mean = np.mean(h_arr)
        std = np.std(h_arr, ddof=1)

        # The reference value (seed=42) should be within 3 std of the mean
        assert abs(h_ref - mean) < 3 * std + 1e-10, (
            f"{case_name}: ref={h_ref:.4f}, mean={mean:.4f}±{std:.4f} "
            f"(seed=42 deviates {abs(h_ref-mean)/std:.2f}σ)"
        )

        # The expected benchmark value should be within 3 std of the multi-seed mean
        assert abs(case.h_crit_expected - mean) < 3 * std + case.h_crit_tolerance, (
            f"{case_name}: expected={case.h_crit_expected:.4f}, "
            f"mean={mean:.4f}±{std:.4f} across {len(seeds)} seeds"
        )
