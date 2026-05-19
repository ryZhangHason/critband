Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install critband

Or using uv:

.. code-block:: bash

   uv add critband


Basic Usage
-----------

The simplest use case: test whether your data is bimodal.

.. code-block:: python

   import numpy as np
   from critband import critical_bandwidth

   # Bimodal data
   x = np.concatenate([
       np.random.normal(-2, 0.5, 200),
       np.random.normal( 2, 0.5, 200),
   ])

   h_crit, success = critical_bandwidth(x)
   print(f"Critical bandwidth: {h_crit:.4f}")
   print(f"Converged: {success}")


Reading Data from Files
-----------------------

critband can read numerical data from 9 file formats without needing to learn
separate tools for each format.

.. code-block:: python

   from critband.io import read_data

   # Auto-detect — just point at any supported file
   x = read_data("measurements.csv")
   x = read_data("survey.xlsx", sheet="Sheet1")
   x = read_data("results.pdf", column="Score")

Supported formats:

- CSV (built-in ``csv``)
- TSV/TXT (built-in)
- JSON (built-in ``json``)
- Markdown tables (built-in parser)
- HTML tables (built-in ``html.parser``)
- XLSX (``openpyxl``)
- XLS (``xlrd``)
- DOCX (``python-docx``)
- PDF (``pdfplumber``)


Full Pipeline Example
---------------------

.. code-block:: python

   import numpy as np
   from critband import (
       critical_bandwidth,
       find_modes,
       bimodality_strength,
       excess_mass,
       find_trough,
       detect_components,
   )

   # Generate bimodal data
   x = np.concatenate([
       np.random.normal(-3, 0.5, 300),
       np.random.normal( 3, 0.5, 300),
   ])

   # 1. Critical bandwidth
   h_crit, success = critical_bandwidth(x)
   print(f"h_crit = {h_crit:.4f}")

   # 2. Interpretable strength score
   s = bimodality_strength(x)
   print(f"Strength: {s.strength} (score={s.strength_score:.2f})")

   # 3. Find KDE modes at critical bandwidth
   modes = find_modes(x, h_crit)
   print(f"Detected {len(modes.modes)} modes")

   # 4. Locate the trough (valley)
   trough = find_trough(x, h_crit)
   print(f"Trough at x = {trough:.4f}")

   # 5. Decompose into Gaussian components
   dec = detect_components(x)
   print(f"Component 1: μ={dec.component1.mean:.2f}, "
         f"σ={dec.component1.std:.2f}, w={dec.component1.weight:.2f}")
   print(f"Component 2: μ={dec.component2.mean:.2f}, "
         f"σ={dec.component2.std:.2f}, w={dec.component2.weight:.2f}")

   # 6. Excess mass multimodality test
   e = excess_mass(x, n_boot=199)
   print(f"Estimated modes: {e.n_modes_estimated}")

   # 7. k-mode detection (e.g., trimodal)
   trimodal = np.concatenate([
       np.random.normal(-4, 0.3, 150),
       np.random.normal( 0, 0.3, 150),
       np.random.normal( 4, 0.3, 150),
   ])
   h_k3, ok = critical_bandwidth(trimodal, k=3)
   print(f"Trimodal critical bandwidth (k=3): {h_k3:.4f}")
