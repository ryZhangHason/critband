critband documentation
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api_reference
   r_comparison
   development


What is critband?
-------------

**critband** is a Python package for detecting and analyzing bimodal (and multimodal)
distributions using the **critical bandwidth** method in kernel density estimation
(KDE). The critical bandwidth :math:`h_{\text{crit}}` is the smallest bandwidth
at which the KDE transitions from :math:`k` modes to :math:`k-1` modes — a
well-established statistical test introduced by Silverman (1981).

Key features:

- **Critical bandwidth solver** — finds the transition point for any
  :math:`k \ge 2` via binary search, Brent's method, or automatic selection
- **Bootstrap inference** — confidence intervals and Silverman's bootstrap
  :math:`p`-value for modality testing
- **Mode detection** — :func:`find_modes` identifies all KDE modes at a given
  bandwidth; :func:`detect_components` decomposes a bimodal distribution into
  Gaussian components
- **Modality tests** — :func:`dip_test` (Hartigan & Hartigan 1985),
  :func:`silverman_test` (Silverman 1981), :func:`excess_mass`
  (Müller & Sawitzki 1991)
- **Multi-format I/O** — :mod:`critband.io` reads numerical data from 9 file formats
  (CSV, TSV, JSON, Markdown, HTML, XLSX, XLS, DOCX, PDF) through a unified API
- **Web-ready** — runs in-browser via Pyodide (WebAssembly) at
  https://qhwangantoneva.github.io/critband/

Quick start:

.. code-block:: python

   import numpy as np
   from critband import critical_bandwidth

   x = np.concatenate([np.random.normal(-2, 0.5, 200),
                        np.random.normal( 2, 0.5, 200)])
   h_crit, success = critical_bandwidth(x)
   print(f"Critical bandwidth: {h_crit:.4f}")  # ~0.97
   print(f"Converged: {success}")              # True


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
