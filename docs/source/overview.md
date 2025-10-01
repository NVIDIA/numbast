# Overview

Numbast generates Python bindings for CUDA C++ libraries and utilities so they can be used from Python with Numba CUDA.

## What Numbast does

- Parse CUDA C++ declarations (including templates) from headers
- Generate Numba-compatible extension types and functions
- Integrate the generated bindings into Python APIs in the CUDA ecosystem

## How it works

Numbast comes with two components: `ast_canopy` and `numbast`.

- `ast_canopy` is the clangTooling binding that parses the CUDA C++ header for declaration information
- `numbast` is the Numba-CUDA binding generator, the consumer of `ast_canopy`'s output

## Modes of operation

Numbast supports two complementary workflows.

### 1. Static binding generation
- Produces a standalone Python binding module that depends only on Numba CUDA.
- The generated module can be distributed and imported without requiring Numbast at runtime.
- Best for library authors who want to avoid adding extra build/runtime dependencies for end users.
- See: [Static binding generation](static.md)

### 2. Dynamic binding generation
- Runs in an environment where the CUDA C++ library and Numbast are available.
- Numbast loads headers, generates bindings, and registers them with Numba at runtime.
- Best for exploratory users who want quick access to a C++ library without a separate build step.
- See: [Dynamic binding generation](dynamic.md)

## Environment assumptions

Both static and dynamic workflows require a correctly configured CUDA environment when kernels are launched. The assumption is enforced at different times:

- Static generation: the environment used to generate bindings must be compatible with the target runtime environment (ideally the same CUDA version, or at least backward compatible).
- Dynamic generation: generation and execution happen in the same environment; headers located at generation time should match the runtime CUDA installation.

Numbast relies on `numba.cuda` (via `cuda_paths.py`) to locate the CUDA toolkit and headers.

## API reference

See the full API reference for `ast_canopy` and `numbast`:

- [API reference](api_reference.rst)
