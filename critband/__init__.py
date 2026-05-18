"""
critband: Python package for calculating critical bandwidth for bimodal distributions

This package implements algorithms to detect the critical bandwidth in kernel density
estimation, which is the smallest bandwidth where the density estimate remains unimodal.
"""

from .bandwidth import (
    BimodalDecomposition,
    BimodalityStrength,
    Component,
    DipTestResult,
    ExcessMassResult,
    Mode,
    ModeResult,
    bimodality_strength,
    critical_bandwidth,
    detect_components,
    dip_test,
    excess_mass,
    find_modes,
    find_trough,
    gaussian_kde,
    gaussian_kde_fft,
    silverman_bandwidth,
)
from .bootstrap import (
    BootstrapResult,
    ModeTestResult,
    SilvermanTestResult,
    bootstrap_critical_bandwidth,
    modetest,
    silverman_test,
)

__version__ = "0.1.0"
__all__ = [
    "silverman_bandwidth",
    "critical_bandwidth",
    "gaussian_kde",
    "gaussian_kde_fft",
    "find_trough",
    "find_modes",
    "detect_components",
    "Component",
    "Mode",
    "ModeResult",
    "BimodalDecomposition",
    "bimodality_strength",
    "BimodalityStrength",
    "dip_test",
    "DipTestResult",
    "excess_mass",
    "ExcessMassResult",
    "bootstrap_critical_bandwidth",
    "BootstrapResult",
    "ModeTestResult",
    "SilvermanTestResult",
    "modetest",
    "silverman_test",
]
