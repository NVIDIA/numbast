FAQ
===

CUDA headers and include paths
------------------------------

**Why does AST Canopy need to find CUDA C/C++ headers?**

AST Canopy uses Clang's CUDA mode to parse the CUDA C++ header. This mode requires the CUDA directory passed
via the `--cuda-path` flag, and the CUDA include directories passed via the `-I` flag.

**How does AST Canopy find CUDA C/C++ headers?**

AST Canopy relies on `cuda.pathfinder <https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/>`_ to
locate the CUDA Toolkit header directory (e.g., the folder containing ``cuda.h``). Internally this uses
``cuda.pathfinder.find_nvidia_header_directory()``, which probes several common locations and environment
variables so you usually don't need to configure anything.

What is searched (high level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a high level overview of the search paths, for more details, please refer to the `cuda.pathfinder documentation
<https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/>`_.

- **Pip installed CUDA Toolkit**: If you installed the CUDA Toolkit via pip (e.g., ``cuda-toolkit[cudart,nvcc]``),
  headers are typically in ``site-packages/nvidia/``.
- **Conda environments**: If you installed the Toolkit via Conda (e.g., ``cudatoolkit``), headers are typically in
  ``$CONDA_PREFIX/include``.
- **Explicit environment variables**: If ``CUDA_HOME`` or ``CUDA_PATH`` is set, their ``include`` subdirectory is
  used (e.g., ``$CUDA_HOME/include``).

Conda vs. pip nuances
^^^^^^^^^^^^^^^^^^^^^

- **pip (runtime-focused)**: Packages like ``cuda-toolkit[cudart,nvcc]`` ship the CUDA headers. But they have different
  layouts in either CUDA 12 or CUDA 13. See the next section for more details.
- **Conda (recommended for headers)**: The ``cuda-toolkit`` package commonly includes header files, so
  ``cuda.pathfinder`` will find them in ``$CONDA_PREFIX/include``.

.. note::
    We recommend using only one package management system for a single CUDA installation.

CUDA 12 vs. CUDA 13 considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Main difference between CUDA 12 and CUDA 13 is the layout of headers in pip installed CUDA Toolkit.

**Runtime headers**:

- **CUDA 12**: Headers are found in ``site-packages/nvidia/cuda_runtime/include``
- **CUDA 13**: Headers are found in ``site-packages/nvidia/cu13/include``

**CCCL headers**:

- **CUDA 12**: Headers are found in ``site-packages/nvidia/cuda_cccl/include``
- **CUDA 13**: Headers are found in ``site-packages/nvidia/cu13/include/cccl``

.. note::
    For CUDA 12, since the site-package directory is non-standard, clang++ is unable to use the site-package directory
    as `--cuda-path`. Instead, please install the system-wide CUDA Toolkit to corresponding version and set
    ``CUDA_HOME`` to the system-wide Toolkit root.
