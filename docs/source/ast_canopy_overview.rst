AST Canopy overview
===================

``ast_canopy`` parses CUDA C++ headers using LLVM's clang tooling and produces
a compact representation of top-level declarations for downstream consumption
by Numbast.

What it provides
----------------

- Python API to parse a translation unit and retain a subset of files
- Serialization of declarations (records, functions, templates, typedefs)
- Utilities for mapping C++ semantics to Python-friendly constructs

Primary API
-----------

.. code-block:: python

   import ast_canopy
   ast_canopy.parse_declarations_from_source(
       source_file_path,
       files_to_retain,
       compute_capability,
       cccl_root=None,
       cudatoolkit_include_dir=None,
       cxx_standard="c++17",
   )

Build and tests
---------------

See :doc:`install` for environment setup. To build and test from the repository root:

.. code-block:: bash

   bash ast_canopy/build.sh
   pytest ast_canopy/

Folder structure
----------------

- ``cpp``: C++ library that traverses clang AST (``libastcanopy``)
- ``ast_canopy``: Python bindings and high-level API
- ``tests``: Unit tests for bindings and API

Notes
-----

- ``libastcanopy`` builds independently with CMake and is used via dynamic linking from Python
- The Python package uses ``scikit-build-core`` and ``pybind11``
