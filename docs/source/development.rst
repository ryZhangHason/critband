Development
===========

.. highlight:: bash

Getting Started
---------------

Clone the repository and install development dependencies:

.. code-block:: bash

   git clone https://github.com/ryZhangHason/Polarization-CBW.git
   cd Polarization-CBW
   uv sync --group dev


Running Tests
-------------

.. code-block:: bash

   # Run all tests
   uv run python -m pytest tests/ -v

   # With coverage
   uv run python -m pytest tests/ --cov


Linting and Formatting
----------------------

.. code-block:: bash

   uv run ruff check .
   uv run ruff format .


Building Documentation
----------------------

.. code-block:: bash

   cd docs
   uv run sphinx-build -b html source/ build/
   # Open docs/build/index.html in your browser


Contributing
------------

We welcome contributions! Please see `CONTRIBUTING.md <https://github.com/ryZhangHason/Polarization-CBW/blob/main/CONTRIBUTING.md>`_ on GitHub for guidelines.
