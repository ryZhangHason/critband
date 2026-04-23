"""
Unit tests for bandwidth calculation functions.
"""

import numpy as np
from scipy import integrate
import pytest
from critical_bandwidth import silverman_bandwidth, critical_bandwidth, gaussian_kde


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
        
        # Should return that data is already unimodal at minimum bandwidth
        assert success is False
    
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
        assert 0.3 < h_crit < 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])