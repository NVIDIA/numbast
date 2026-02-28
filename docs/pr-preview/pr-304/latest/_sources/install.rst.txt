Install
-------

Prebuilt packages are available on both PyPI and Conda. For most users, we recommend installing
from PyPI. For development, we recommend installing from source with Pixi.

Install via PyPI (recommended)
------------------------------

.. code-block:: bash

  pip install numbast


Install via Conda (alternative)
-------------------------------

.. code-block:: bash

  conda install numbast


From source with Pixi (recommended for development)
---------------------------------------------------

Numbast depends on LLVM's libClangTooling. For an easy development setup, use the repository
``pixi.toml``, which defines two CUDA test groups:

- ``test-cu12`` (CUDA 12.9)
- ``test-cu13`` (CUDA 13.0)

.. code-block:: bash

  # from repo root, install one environment
  pixi install -e test-cu12
  # or
  pixi install -e test-cu13

  # build and install local packages in the selected environment
  pixi run -e test-cu12 build-ast-canopy
  pixi run -e test-cu12 install-numbast
  pixi run -e test-cu12 install-numbast-extensions


Validate the installation (optional)
------------------------------------

.. code-block:: bash

  pixi run -e test-cu12 test

Replace ``test-cu12`` with ``test-cu13`` if you are testing against CUDA 13.

.. note::
  If you see errors like "cannot find header 'cuda.h'", please refer to :doc:`FAQ <faq>` for more details.

Building Documentation
----------------------

Dependencies
^^^^^^^^^^^^

Use Pixi to ensure consistent versions. In ``pixi.toml``, the ``docs`` feature includes:

- sphinx
- sphinx-copybutton
- nvidia-sphinx-theme

You also need Python and a working environment to import ``numbast`` if you want versioned builds
to reflect the installed package version (optional). You can enter one with
``pixi shell -e test-cu13``.


Build Steps
^^^^^^^^^^^

- Recommended: build all versions listed in ``docs/nv-versions.json`` via the Pixi task from
  the repository root:

  .. code-block:: bash

    pixi run -e test-cu13 build-docs

- Build only the latest version:

  .. code-block:: bash

    pixi run -e test-cu13 build-docs latest-only

  Replace ``test-cu13`` with ``test-cu12`` if needed.

- Optional fallback (if you are already in a prepared shell environment):

  .. code-block:: bash

    cd docs
    ./build_docs.sh
    # or
    ./build_docs.sh latest-only

Artifacts are generated under:

- ``docs/build/html/latest``
- ``docs/build/html/<version>`` where ``<version>`` comes from
  ``SPHINX_NUMBAST_VER`` or the installed ``numbast`` package version.

Notes
^^^^^

- The build script sets ``SPHINX_NUMBAST_VER`` from the installed ``numbast`` package version unless
  ``SPHINX_NUMBAST_VER`` is provided explicitly.
- Output also copies ``versions.json`` and ``nv-versions.json``, and creates a redirect ``index.html``.
