CUDA-Oxide device bindings
===========================

The ``cuda-oxide`` backend generates a Rust ``sys`` module for C-linkage CUDA
device functions. It parses the same header profile and reuses the same
selection and naming configuration as Numbast's static Numba backend, but the
renderer has no Numba import or runtime dependency.

Round 1 contract
----------------

The initial backend accepts externally visible, non-variadic ``extern "C"``
``__device__`` and ``__host__ __device__`` free functions. It supports C
scalars, nested pointers with per-level ``const`` qualification, arrays behind
pointers, typedefs, enums, opaque handles, the CUDA half/bfloat16 and
``double2`` storage ABIs, and POD records used behind pointers.

CUDA-Oxide's pre-Blackwell NVVM path cannot call extern functions that pass
sub-32-bit integers, Boolean values, CUDA half, or bfloat16 by value. Numbast
still emits the complete selected C surface, uses storage-compatible types for
pointer APIs, lists every affected symbol in ``compatibility`` in the manifest,
and marks whether the configured architecture supports the whole surface.
Calling those functions requires ``sm_100`` or newer. On that modern path,
CUDA ``__half`` maps to Rust's nightly ``f16`` and the consuming crate must
enable ``#![feature(f16)]``. This restriction comes from CUDA-Oxide's selected
NVVM dialect; no hidden ABI shim is generated to bypass it.

It intentionally rejects C++ linkage, overloads, references, variadics,
by-value records, and unknown ABI types. Function templates and class templates
are recorded as Round 1 exclusions. Argument-intent adapters are a later
feature; a nonempty ``Function Argument Intents`` mapping is rejected rather
than silently ignored, and every generated raw call is ``unsafe``.

The output contains direct CUDA-Oxide declarations:

.. code-block:: rust

  use cuda_device::device;

  #[device]
  unsafe extern "C" {
      pub fn library_add(lhs: i32, rhs: i32) -> i32;
  }

  pub use self::library_add as add;

There is no generated CUDA C or C++ shim. CUDA-Oxide compiles callers to device
IR and links the external LTOIR named in the manifest.

Configuration
-------------

Select the backend with ``Backend: cuda-oxide`` and add a ``CUDA Oxide``
section to the existing Numbast YAML shape:

.. code-block:: yaml

  Entry Point: include/library_device.h
  File List: [include/library_device.h]
  GPU Arch: [sm_90]
  Clang Include Paths: [include]
  Predefined Macros: [LIBRARY_BUILD_DEVICE_LTOIR]
  Exclude:
    Function: [library_internal_helper]
  Skip Prefix: library_detail_
  API Prefix Removal:
    Function: [library_]
  Output Name: library_device.rs

  Backend: cuda-oxide
  CUDA Oxide:
    LTOIR Inputs: [lib/library_device.ltoir]
    Manifest Name: library_device.manifest.json
    Symbol Inventory: build/library_device_api.symbols
    Type Aliases:
      library_handle_t: unsigned long long
    Constants:
      LIBRARY_DEFAULT_HANDLE: {Type: library_handle_t, Value: 0}

``Type Aliases`` are C-to-C declarations, not arbitrary Rust snippets.
``Constants`` accept typed integer or Boolean literals for definitions absent
from the AST, such as preprocessor macros.

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

The ``compatibility`` object also identifies declarations that require the
modern ``sm_100+`` CUDA-Oxide NVVM path because they pass sub-32-bit values by
value.

The LTOIR producer and Rust build should consume the same architecture,
headers, macros, and library version. Numbast records link inputs; CUDA-Oxide
continues to own Rust device compilation and final linking.

Commands and examples
---------------------

Use the dedicated lightweight command (which does not import Numba):

.. code-block:: bash

  numbast-cuda-oxide --cfg-path config.yaml --output-dir generated

``examples/cuda_oxide_c_device`` builds a small C device library as LTOIR,
generates and verifies bindings, compiles a Rust kernel, links both LTOIR
inputs, and executes the result on a GPU. ``examples/nvshmem_cuda_oxide``
contains the complete selected NVSHMEM configuration profile and migration
notes.
