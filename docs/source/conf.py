"""Sphinx configuration for critband documentation."""

import sys
from pathlib import Path

# Add project root to path so autodoc can find the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "critband"
copyright = "2026, Ruiyu Zhang"
author = "Ruiyu Zhang"
release = "0.2.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_attr_annotations = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_title = "critband — Critical Bandwidth for Multimodal Distributions"
html_theme_options = {
    "source_repository": "https://github.com/ryZhangHason/critband/",
    "source_branch": "main",
    "source_directory": "docs/source",
}
