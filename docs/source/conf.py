import os
import sys

# -- Project information -----------------------------------------------------

project = "Numbast"
author = "NVIDIA"

# Use environment to avoid importing the package during docs build
release = os.environ.get("SPHINX_NUMBAST_VER", "latest")

# Ensure local packages are importable for autodoc/autosummary
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "numbast", "src")
for path in (SRC_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

html_baseurl = "docs"
html_theme = "nvidia_sphinx_theme"
html_logo = "_static/logo-light-mode.png"

html_static_path = ["_static"]

# Skip copying prompts when using copybutton
copybutton_exclude = ".linenos, .gp"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
