"""
Unit tests for bandwidth calculation functions.
"""

import numpy as np
from scipy import integrate
import pytest
from pola import silverman_bandwidth, critical_bandwidth, gaussian_kde
from pola.bandwidth import count_modes


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
        x = np.concatenate([
            np.random.normal(-1, 0.2, 100),
            np.random.normal(1, 0.2, 100)
        ])
        
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
        x = np.concatenate([
            np.random.normal(-2, 0.3, 200),
            np.random.normal(2, 0.3, 200),
        ])
        h1, _ = critical_bandwidth(x)
        x_noisy = x + np.random.normal(0, 1e-6, len(x))
        h2, _ = critical_bandwidth(x_noisy)
        assert abs(h1 - h2) < 0.1

    def test_tolerance_does_not_harm_convergence(self):
        np.random.seed(42)
        x = np.concatenate([
            np.random.normal(-2, 0.3, 200),
            np.random.normal(2, 0.3, 200),
        ])
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
        expected = np.array([
            np.sum(np.exp(-(xi - x)**2 / (2 * h**2)))
            / (len(x) * h * np.sqrt(2 * np.pi))
            for xi in grid
        ])
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