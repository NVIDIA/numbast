Numba-CUDA-MLIR support
=======================

Numbast can generate bindings for both the default
`Numba-CUDA <https://nvidia.github.io/numba-cuda/>`__ backend and the
experimental
`numba-cuda-mlir <https://nvidia.github.io/numba-cuda-mlir/latest/>`__
backend. The CUDA C++ parsing workflow is the same for both backends; the
differences are the Python import namespace, runtime dependency, and static
binding configuration.

Backend differences
-------------------

.. list-table::
   :header-rows: 1

   * - Area
     - Numba-CUDA
     - numba-cuda-mlir
   * - Dynamic Numbast imports
     - Import binding helpers from top-level ``numbast``.
     - Import binding helpers from ``numbast.experimental.mlir``.
   * - Static binding generation
     - Use the default static generator configuration.
     - Add ``MLIR Backend: true`` to the static binding config.
   * - Generated module target
     - Generated bindings target the ``numba-cuda`` runtime.
     - Generated bindings target the ``numba-cuda-mlir`` runtime.
   * - Dependency setup
     - Install Numbast with the default ``numba-cuda`` dependency.
     - For dynamic binding generation, install Numbast with the ``mlir`` extra
       so ``numba-cuda-mlir`` is available.

Dynamic binding generation
--------------------------

If you intend to use dynamic binding generation with ``numba-cuda-mlir``,
install Numbast with the ``mlir`` extra:

.. code-block:: bash

   pip install "numbast[mlir]"

For dynamic binding generation, change imports from top-level ``numbast`` to
``numbast.experimental.mlir``. The Numbast helper names remain the same.

Before:

.. code-block:: python

   from numbast import bind_cxx_function, bind_cxx_struct, MemoryShimWriter

After:

.. code-block:: python

   from numbast.experimental.mlir import (
       bind_cxx_function,
       bind_cxx_struct,
       MemoryShimWriter,
   )

Use ``numba_cuda_mlir`` types, models, and CUDA entry points with the generated
bindings.

.. code-block:: python

   from numba_cuda_mlir import cuda, types
   from numba_cuda_mlir.models import PrimitiveModel


Static binding generation
-------------------------

For static binding generation, add the ``MLIR Backend`` entry to the same
config file you use with the default Numbast CLI.

.. code-block:: yaml

   Entry Point: /path/to/library/header.hpp
   File List:
     - /path/to/library/header.hpp
   GPU Arch: ["sm_80"]

   MLIR Backend: true

Then run the standard Numbast static binding generator:

.. code-block:: bash

   python -m numbast --cfg-path config.yml --output-dir ./output

With ``MLIR Backend: true``, the CLI routes generation through
``numbast.experimental.mlir`` and writes a module for ``numba-cuda-mlir``.

Notes
-----

- The MLIR backend is experimental and is not re-exported from top-level
  ``numbast``.
- The environment that generates or imports MLIR-backed bindings must provide
  ``numba_cuda_mlir``.
- Backend-specific static config keys that mention MLIR require
  ``MLIR Backend: true``.
