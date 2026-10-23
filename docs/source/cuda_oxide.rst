CUDA-Oxide device bindings
==========================

The ``cuda-oxide`` backend parses C-linkage CUDA device declarations and emits
an includable Rust module plus a versioned JSON manifest. It reuses the shared
configuration, binding plan, and Rust renderer. Generation does not import Numba
or require a GPU; linking and running the result requires CUDA-Oxide and the
external CUDA LTOIR artifacts.

Configuration and commands
--------------------------

.. code-block:: yaml

   Backend: cuda-oxide
   Entry Point: include/device_api.h
   File List: [include/device_api.h]
   GPU Arch: [sm_90]
   Output Name: python_bindings.py
   CUDA Oxide:
     Output Name: bindings.rs
     LTOIR Inputs: [lib/device_api.ltoir]
     Manifest Name: bindings.manifest.json
     Symbol Inventory: build/device_api.symbols
     Type Aliases:
       library_handle_t: unsigned long long
     Constants:
       LIBRARY_DEFAULT_HANDLE: {Type: library_handle_t, Value: 0}

``Backend: [cuda-oxide]`` also selects the backend. A single invocation generates
one backend. The top-level ``Output Name`` is a Numba setting; Rust uses the
nested output name. Numba-only argument-intent settings are ignored.

.. code-block:: bash

   numbast --cfg-path bindings.yaml --output-dir generated
   # Or invoke the dedicated driver:
   numbast-cuda-oxide --cfg-path bindings.yaml --output-dir generated

The dedicated command also accepts a document without a ``Backend`` selector.
An explicitly selected different backend is rejected. The shared command uses
the backend registry for all selection and imports.

Strict parsing is the default. ``CUDA Oxide.Bypass Parse Errors: true`` enables
recovery; an explicitly supplied ``-noraise true`` or ``-noraise false`` overrides
that setting. Pair recovery with an exact symbol inventory to catch omissions.
The shared ``-fmt`` option applies only to Python output and has no effect on
Rust output.

Outputs and verification
------------------------

The manifest records source and configuration hashes, the parser profile,
architecture, native and Rust symbol names, types, constants, exclusions, and
ordered LTOIR paths. Existing LTOIR inputs are hashed; absent inputs are recorded
as missing at generation time. Numbast does not build or link those inputs.

``Symbol Inventory`` is optional. When supplied, it must contain the exact
selected public API, as one symbol per line or an ``nm``-style listing. Missing,
unexpected, or malformed symbols fail generation before either output is
written. Each output is replaced atomically.

Global scalar ``const`` and ``constexpr`` declarations from retained headers
are emitted automatically when using AST Canopy's global constant collection.
Initializer expressions are evaluated by Clang. Parsed enum members are also
emitted automatically. Their values do not need to be repeated in YAML.

``Type Aliases`` maps C names to C type spellings. ``Constants`` supplies typed
integer or Boolean literals absent from the AST; expressions are rejected.
Generated declarations are raw unsafe device calls. The binding plan enforces
its supported C ABI types and target architecture restrictions.
