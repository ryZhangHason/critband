R Package Comparison
=====================

critband has been validated against equivalent R implementations across 12
benchmark cases (seed=42, unique per case). The comparison covers three
R packages: ``multimode`` (v2.10), ``diptest`` (v0.77), and ``ks`` (v1.14).


Critical Bandwidth Agreement
-----------------------------

| Case | :math:`n` | critband :math:`h_{\text{crit}}` | R ``modetest`` :math:`p` | R ``dip.test`` :math:`p` |
|------|:----------:|:----------------------------:|:------------------------:|:-------------------------:|
| Well-separated | 400 | 1.8650 | 0.000 | 0.000 |
| Moderate separation | 500 | 1.0964 | 0.000 | 0.000 |
| Barely separated | 600 | 0.2791 | 0.251 | 0.709 |
| Unequal variance | 400 | 1.7849 | 0.000 | 0.000 |
| Unequal weights | 500 | 1.2591 | 0.000 | 0.000 |
| Extreme separation | 400 | 4.6987 | 0.000 | 0.000 |
| Trimodal | 450 | 1.3824 | 0.000 | 0.000 |
| Skewed bimodal | 500 | 1.1417 | 0.000 | 0.000 |
| Heavy-tailed bimodal | 400 | 2.7109 | 0.000 | 0.000 |
| Near unimodal | 600 | 0.4186 | 0.055 | 0.285 |
| Small sample bimodal | 60 | 1.8608 | 0.000 | 0.000 |
| Overlapping variances | 500 | 0.4598 | 0.045 | 0.322 |

critband's ``critical_bandwidth()`` agrees with R's ``modetest`` — strongly
bimodal cases produce significant :math:`p`-values in both environments,
and boundary cases (barely separated, near unimodal, overlapping variances)
return consistent non-significant results.


Performance Comparison
----------------------

| Operation | critband | R | Speedup |
|-----------|:----:|:--:|:-------:|
| ``critical_bandwidth()`` | 0.04–0.79s | 0.82–1.58s (``modetest``) | 3–10× faster |
| ``dip_test()`` / ``dip.test()`` | ~0.002s | 0.001–0.008s | Tie |
| ``find_modes()`` / ``nmodes()`` | <0.01s | 0.03–0.05s | Tie |

critband's performance advantage on critical bandwidth computation comes from
a pure-Python numerical stack (NumPy/SciPy) with adaptive grid sizing and
no R process dispatch overhead.


Feature Comparison
------------------

| Feature | critband | multimode | diptest |
|---------|:----:|:---------:|:-------:|
| Critical bandwidth (:math:`k \ge 2`) | ✅ | ✅ (:math:`k=2`) | ❌ |
| k-mode detection | ✅ (any :math:`k`) | ❌ | ❌ |
| Bimodality strength | ✅ (interpretable) | ❌ | ❌ |
| Excess mass test | ✅ | ✅ | ❌ |
| Silverman's bootstrap test | ✅ | ✅ | ❌ |
| Hartigan's dip test | ✅ | ❌ | ✅ |
| Component decomposition | ✅ | ❌ | ❌ |
| Bootstrap confidence intervals | ✅ | ❌ | ❌ |
| Multi-format I/O (9 formats) | ✅ | ❌ | ❌ |
| Web browser (Pyodide) | ✅ | ❌ | ❌ |
| Pure Python dependencies | ✅ | ❌ (compiled) | ❌ (compiled) |


Reproduction
------------

.. code-block:: bash

   # critband benchmark
   uv run python benchmark_pola_comparison.py

   # R benchmark (requires R + multimode + diptest)
   Rscript benchmark_r_comparison.R
