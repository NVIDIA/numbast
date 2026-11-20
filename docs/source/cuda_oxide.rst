CUDA-Oxide device bindings
===========================

The ``cuda-oxide`` backend currently supports the NVSHMEM4Rust device binding
profile. It generates a Rust ``sys`` module for the selected NVSHMEM C-linkage
device API.

NVSHMEM4Rust device profile
---------------------------

The profile generates raw ``unsafe`` externs for selected non-variadic
``__device__`` and ``__host__ __device__`` functions. The bindings preserve the
pointer qualifiers and ABI types used by these functions, including the
half, bfloat16, and ``double2`` storage types. The profile excludes internal
functions and C++-linkage helpers.

The complete profile requires ``sm_100`` or newer because it passes narrow
integers, CUDA half, and bfloat16 by value. Pre-Blackwell CUDA-Oxide can use
pointer-based forms of those types. CUDA ``__half`` maps to Rust's nightly
``f16`` on the modern NVVM path, so the consuming crate must enable
``#![feature(f16)]``.

The generated Rust module contains declarations like this one:

.. code-block:: rust

  use cuda_device::device;

  #[device]
  unsafe extern "C" {
      pub fn nvshmemx_vendor_get_version_info(
          major: *mut i32, minor: *mut i32, patch: *mut i32,
      );
  }

  pub use self::nvshmemx_vendor_get_version_info as vendor_get_version_info;

There is no generated CUDA C or C++ shim. CUDA-Oxide compiles callers to device
IR and links the external LTOIR named in the manifest.

Configuration
-------------

Fill in the placeholders in
``examples/nvshmem_cuda_oxide/numbast.yaml.in`` using the NVSHMEM source and
build that produced the device LTOIR. The template selects
``Backend: cuda-oxide`` and uses the NVSHMEM4Rust entry header.

The generator reads typedefs, enum values, and supported scalar macros from
the retained headers. It records evaluated macros in the manifest with
``source: macro``.

Numbast derives the matching ``__CUDA_ARCH__`` value from ``GPU Arch`` while
parsing. An explicitly configured value must match. This exposes APIs guarded
by device-compilation conditionals without letting the parser profile drift
from the link target.

Symbol parity and manifest
--------------------------

``Symbol Inventory`` is an exact, newline- or ``nm``-style inventory of the
selected public API. Generation fails if a symbol exists on only one side, so
recoverable parser diagnostics cannot silently shrink the bindings. Filter
implementation-only symbols before supplying an inventory.

The versioned JSON manifest records:

- the target architecture and NVPTX C ABI;
- source paths, hashes, include paths, and effective macros;
- ordered LTOIR paths and hashes when the artifacts are present;
- every native, public, and Rust name with normalized C and Rust types;
- generated type and constant metadata;
- every intentional exclusion; and
- exact symbol-verification status.

The ``compatibility`` object records the CUDA-Oxide NVVM dialect selected by
the configured architecture.

The LTOIR producer and Rust build should consume the same architecture,
headers, macros, and library version. Numbast records link inputs; CUDA-Oxide
continues to own Rust device compilation and final linking.

Commands and examples
---------------------

Use the dedicated lightweight command (which does not import Numba):

.. code-block:: bash

  numbast-cuda-oxide --cfg-path config.yaml --output-dir generated

``examples/nvshmem_cuda_oxide`` contains the NVSHMEM4Rust configuration and a
GPU smoke test. ``examples/cuda_oxide_c_device`` demonstrates the same LTOIR
build, binding generation, and linking flow with a small test library.
