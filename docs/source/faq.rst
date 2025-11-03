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


Clang requirements (host headers and resources)
-----------------------------------------------

**What are the minimum host-side requirements to parse CUDA headers?**

AST Canopy drives Clang in CUDA mode. Even for device-only parsing, Clang needs two host-side components:

- libstdc++ C++ headers (for headers like ``<cstdlib>``, ``<cmath>`` used via CUDA wrappers). In principle,
  use the supported libstdc++ versions listed by clang.
- Clang "resource directory" headers (``<resource>/include`` and ``include/cuda_wrappers`` containing
  ``__clang_cuda_runtime_wrapper.h`` and friends).

You do not need to link libstdc++ for device-only parsing, but their headers must be discoverable.

Environment-specific discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use Clang's driver logic (in-process) to compute the same include search paths that ``clang++`` would pass to
``cc1``. How the headers are found depends on your environment:

1. Pip wheel/bare-metal (system toolchain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Install system C++ headers (e.g., ``libstdc++-dev`` on Debian/Ubuntu) and Clang resource headers
  (e.g., ``libclang-common-20-dev``).
- AST Canopy invokes Clang's driver API with your host triple to discover C++ standard library include dirs and system
  C headers. On Linux this is typically libstdc++ by default (e.g., ``/usr/include/c++/<ver>``, multiarch dirs,
  ``/usr/include``) along with the resource includes.

2. Conda environments (conda-forge toolchains)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Conda packages (e.g., ``clangdev`` + ``cxx-compiler``) may include a GCC with its own libstdc++ headers.
- We point the driver to the Conda ``clang++`` (via program path/InstalledDir) so it reads the adjacent config and
  discovers sibling GCC/libstdc++ directories automatically. This reproduces the same include list you see from
  running ``clang++ -###`` inside the environment.

3. Custom Clang binary
~~~~~~~~~~~~~~~~~~~~~~

``clang++`` binary location usually includes Clang config file to instruct the driver to discover host resources. User
may also specify a custom clang++ binary path to driver discovery explicitly.

- Set ``ASTCANOPY_CLANG_BIN`` to a specific ``clang++`` path if you want to direct discovery through a custom
  installation.

Notes
^^^^^

- ``-resource-dir`` only affects Clang's builtin headers (including ``cuda_wrappers``); it does not provide C++
  standard library headers.
- AST Canopy does not enforce a specific standard library: the driver chooses based on the toolchain and flags. To use
  libc++, ensure its headers are installed (e.g., ``/usr/include/c++/v1`` or Conda ``libcxx-devel``) and pass
  ``-stdlib=libc++``; otherwise libstdc++ is typically selected by default on Linux.
- AST Canopy is linked against Clang 20. For host resources, please use corresponding Clang version.
