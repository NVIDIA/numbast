Install
=======

Prebuilt Conda packages are available. For development, we recommend installing from source with
Pixi.

Install via Conda
-----------------

.. code-block:: bash

  conda install Numbast


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

- Build all versions listed in ``docs/versions.json``:

  .. code-block:: bash

    cd docs
    ./build_docs.sh

- Build only the latest version:

  .. code-block:: bash

    cd docs
    ./build_docs.sh latest-only

Artifacts are generated under:

- ``docs/build/html/latest``
- ``docs/build/html/<version>`` where ``<version>`` comes from ``SPHINX_NUMBAST_VER`` or detected package/version files.

Notes
^^^^^

- The build script sets ``SPHINX_NUMBAST_VER`` automatically from the installed ``numbast`` package version,
  ``numbast/VERSION``, or top-level ``VERSION``. If none is found, it uses ``latest``.
- Output also copies ``versions.json`` and creates a redirect ``index.html`` for convenience.
