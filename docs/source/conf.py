import os
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[1]
sys.path.insert(0, str(SOURCE_DIR / "_ext"))

from static_binding_schema_doc import (  # noqa: E402
    generate_static_binding_schema_reference,
)

STATIC_BINDING_SCHEMA_REPO_PATH = (
    "numbast/src/numbast/tools/static_binding_generator.schema.yaml"
)
STATIC_BINDING_SCHEMA_PATH = REPO_ROOT / STATIC_BINDING_SCHEMA_REPO_PATH
STATIC_BINDING_SCHEMA_RST_PATH = (
    SOURCE_DIR / "generated" / "static_binding_schema_reference.rst"
)
generate_static_binding_schema_reference(
    schema_path=STATIC_BINDING_SCHEMA_PATH,
    output_path=STATIC_BINDING_SCHEMA_RST_PATH,
    schema_repo_path=STATIC_BINDING_SCHEMA_REPO_PATH,
)

# -- Project information -----------------------------------------------------

project = "Numbast"
copyright = "2023-2025, NVIDIA"
author = "NVIDIA"

# Use environment to avoid importing the package during docs build
release = os.environ.get("SPHINX_NUMBAST_VER", "latest")

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
html_theme_options = {
    "switcher": {
        "json_url": "https://nvidia.github.io/numbast/nv-versions.json",
        "version_match": release,
    },
    "navbar_center": ["version-switcher", "navbar-nav"],
}

html_static_path = ["_static"]

# Skip copying prompts when using copybutton
copybutton_exclude = ".linenos, .gp"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
