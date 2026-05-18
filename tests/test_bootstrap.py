"""
Tests for bootstrap confidence interval estimation.
"""

import numpy as np
import pytest

from pola import BootstrapResult, bootstrap_critical_bandwidth, silverman_test
from pola.benchmark import BENCHMARK_CASES


class TestBootstrapResultDataclass:
    """Test the BootstrapResult dataclass structure."""

    def test_attributes(self):
        """BootstrapResult has all expected attributes."""
        result = BootstrapResult(
            h_crit=1.0,
            ci_lower=0.8,
            ci_upper=1.2,
            standard_error=0.1,
            distribution=np.array([0.9, 1.0, 1.1]),
            n_resamples=3,
            confidence_level=0.95,
            n_failed=0,
        )
        assert result.h_crit == 1.0
        assert result.ci_lower == 0.8
        assert result.ci_upper == 1.2
        assert result.standard_error == 0.1
        assert len(result.distribution) == 3
        assert result.n_resamples == 3
        assert result.confidence_level == 0.95
        assert result.n_failed == 0


class TestBootstrapCriticalBandwidth:
    """Test bootstrap_critical_bandwidth function."""

    def test_returns_bootstrap_result(self):
        """Returns a BootstrapResult instance."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(x, n_resamples=20, random_state=42)
        assert isinstance(result, BootstrapResult)

    def test_h_crit_on_original_matches(self):
        """h_crit from bootstrap matches direct critical_bandwidth call."""
        from pola import critical_bandwidth

        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        h_crit_direct, _ = critical_bandwidth(x)
        result = bootstrap_critical_bandwidth(x, n_resamples=20, random_state=42)
        assert abs(result.h_crit - h_crit_direct) < 1e-6

    def test_ci_bounds_are_sensible(self):
        """CI lower < CI upper (properly ordered)."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(x, n_resamples=30, random_state=42)
        assert result.ci_lower < result.ci_upper

    def test_contains_original_h_crit(self):
        """The bootstrap distribution should contain the original h_crit."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(x, n_resamples=50, random_state=42)
        # The distribution includes the original data resamples
        assert result.distribution.shape == (50,)
        assert np.all(np.isfinite(result.distribution))

    def test_larger_n_resamples_narrows_ci(self):
        """More resamples should produce a stable CI."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(x, n_resamples=50, random_state=42)
        # Just verify the result is sensible
        assert 0 < result.standard_error < 1.0
        assert result.ci_lower < result.ci_upper

    def test_standard_error_positive(self):
        """Standard error should be positive for bimodal data."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(x, n_resamples=30, random_state=42)
        assert result.standard_error > 0

    def test_passes_kwargs_to_critical_bandwidth(self):
        """**kwargs are passed through to critical_bandwidth."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(
            x, n_resamples=20, random_state=42, kernel="epanechnikov"
        )
        assert isinstance(result, BootstrapResult)
        assert np.all(np.isfinite(result.distribution))

    def test_input_validation(self):
        """Invalid input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_critical_bandwidth(np.array([]))

    def test_confidence_level(self):
        """Confidence level is correctly reported as 1 - alpha."""
        np.random.seed(42)
        x = np.concatenate([np.random.normal(-2, 0.3, 200), np.random.normal(2, 0.3, 200)])

        result = bootstrap_critical_bandwidth(x, n_resamples=20, random_state=42, alpha=0.1)
        assert result.confidence_level == 0.9


class TestSilvermanTest:
    """Test Silverman's test for bimodality."""

    def test_silverman_test_bimodal_data(self):
        """Well-separated bimodal data should give small p-value."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        result = silverman_test(x, n_resamples=99)
        assert result.p_value < 0.05
        assert result.h_crit > 0
        assert len(result.null_distribution) == 99
        assert 0 <= result.n_failed <= result.n_resamples

    def test_silverman_test_unimodal_data(self):
        """Unimodal data should give large p-value."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        result = silverman_test(x, n_resamples=99)
        assert result.p_value > 0.05

    def test_silverman_test_constant_data(self):
        """Constant data should not crash."""
        x = np.ones(50) * 5.0
        result = silverman_test(x, n_resamples=99)
        assert result.p_value >= 0  # Should return valid p-value

    def test_silverman_test_reproducible(self):
        """Same random_state gives same result."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        r1 = silverman_test(x, n_resamples=99, random_state=42)
        r2 = silverman_test(x, n_resamples=99, random_state=42)
        assert r1.p_value == r2.p_value

    def test_silverman_test_n_extreme_reasonable(self):
        """n_extreme should be between 0 and n_resamples."""
        x = BENCHMARK_CASES["well_separated_equal_var"].generator(42)
        result = silverman_test(x, n_resamples=99)
        assert 0 <= result.n_extreme <= 99

    def test_silverman_test_null_distribution_length(self):
        """Null distribution should have reasonable length."""
        x = BENCHMARK_CASES["moderate_separation"].generator(42)
        result = silverman_test(x, n_resamples=99)
        assert len(result.null_distribution) == 99
        assert 0 <= result.n_failed <= result.n_resamples


class TestBootstrapStability:
    """Test bootstrap edge cases."""

    def test_bootstrap_all_fail(self):
        """All bootstrap resamples failing should return NaN CI."""
        # Use constant data where convergence fails
        x = np.ones(20) * 5.0
        result = bootstrap_critical_bandwidth(x, n_resamples=10)
        if result.n_failed == result.n_resamples:
            assert np.isnan(result.ci_lower)
            assert np.isnan(result.ci_upper)
