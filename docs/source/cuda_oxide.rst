CUDA-Oxide device bindings
===========================

The ``cuda-oxide`` backend generates a Rust ``sys`` module and a link manifest
from CUDA declarations. It uses Numbast's shared declaration selection and
naming configuration. The renderer runs without a Numba dependency.

Supported declarations and types
--------------------------------

The backend generates direct ``unsafe`` Rust externs for externally visible,
non-variadic C-linkage ``__device__`` and ``__host__ __device__`` free functions.
It supports C scalars, nested pointers with per-level ``const`` qualification,
fixed-size arrays behind pointers, typedefs used by selected signatures, enums,
opaque handles, and POD records behind pointers. Supported CUDA ABI types
include half, bfloat16, and ``double2`` storage.

C++ linkage, overloads, references, variadics, by-value records, and
unrecognized ABI types are unsupported. The manifest records function and
class templates as exclusions. The backend ignores the Numba
``Function Argument Intents`` setting.

Target requirements
-------------------

On targets below ``sm_100``, Numbast rejects extern signatures that pass
sub-32-bit integers, Boolean values, CUDA half, or bfloat16 by value. The
legacy NVVM path supports these types behind pointers. On the modern path,
CUDA ``__half`` maps to Rust's nightly ``f16``, which requires
``#![feature(f16)]`` in the consuming crate.

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

  Backend: cuda-oxide
  CUDA Oxide:
    Output Name: library_device.rs
    LTOIR Inputs: [lib/library_device.ltoir]
    Manifest Name: library_device.manifest.json
    Symbol Inventory: build/library_device_api.symbols
    Type Aliases:
      library_handle_t: unsigned long long

The same document can be used in separate Numba and CUDA-Oxide invocations.
The top-level ``Output Name`` names the Python output; ``CUDA Oxide.Output Name``
names the Rust output. Adding a ``CUDA Oxide`` section alone does not select
the Rust backend.

``Type Aliases`` are C-to-C declarations, not arbitrary Rust snippets.
The generator emits evaluated object-like macros from the retained headers
when their scalar types can be represented in Rust. The manifest records them
with ``source: macro``. Optional ``Constants`` entries provide typed integer or
Boolean literals and take precedence over a header macro with the same name.

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

``examples/cuda_oxide_c_device`` demonstrates the full workflow with a small
C device library, from building LTOIR and verifying symbols to linking and
running a Rust kernel on a GPU.
