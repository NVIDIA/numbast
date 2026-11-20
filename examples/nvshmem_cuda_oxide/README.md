# NVSHMEM4Rust device profile

This directory contains the library-specific configuration needed to replace
NVSHMEM4Rust's handwritten device renderer with Numbast's CUDA-Oxide backend.
The Numbast generator owns declaration parsing, C ABI normalization, Rust name
handling, extern rendering, and the link manifest. NVSHMEM retains ownership
of its header profile, LTOIR, and symbol inventory.

Configure the placeholders in `numbast.yaml.in` from the same NVSHMEM and CUDA
build used to create the device LTOIR, then run:

```bash
numbast-cuda-oxide \
  --cfg-path /path/to/configured/numbast.yaml \
  --output-dir /path/to/nvshmem4rust/runtime/src
```

The NVSHMEM4Rust entry point defines `NVSHMEM_BUILD_LTOIR_LIBRARY`. Numbast
derives `__CUDA_ARCH__` from `GPU Arch`, which is required because these
headers guard the device API with that macro. `nvshmemi_` implementation
symbols, the internal `nvshmemi_signal` template, and CUDA's
`atomicAdd_system` helper are intentionally outside the public C API.

Populate `@NVSHMEM_INCLUDED_FUNCTIONS@` with a YAML sequence of native
device symbols from the matching NVSHMEM build. This example extracts public
symbols from retained PTX and prints a JSON array, which is valid YAML:

```bash
awk '$1 == "//" && $2 == ".globl" && $3 ~ /^nvshmemx?_/ {print $3}' \
  nvshmem_device.ptx | sort -u \
  | python -c 'import json, sys; print(json.dumps(sys.stdin.read().splitlines()))'
```

`Include.Function` requires every listed symbol except those removed by
`Exclude.Function` or `Skip Prefix`. On
NVSHMEM commit `b0d9d3`, the audited profile selected 2,730 public C device
functions, 50 `nvshmemi_` exclusions, one configured CUDA-helper exclusion,
and one template exclusion. The count is a validation snapshot, not a
hardcoded generator rule; generation fails if a required declaration is missing.

No C or C++ shim is generated. The resulting Rust module calls native
NVSHMEM C symbols directly, and CUDA-Oxide links the supplied device LTOIR.

NVSHMEM includes `char`, `short`, half, and bfloat16 APIs with sub-32-bit
values passed by value. CUDA-Oxide supports those exact extern ABIs only on its
modern `sm_100+` NVVM path. Numbast therefore requires an `sm_100+` profile to
generate the complete NVSHMEM module. Older targets can bind pointer-based and
other compatible APIs only after the incompatible by-value declarations are
explicitly excluded; Numbast does not silently widen or shim their ABI.

## Opt-in device execution

The `e2e` crate is a maintained GPU smoke test for a configured profile. It
generates the full module, compiles a Rust kernel that calls the generated
prefix-stripped `vendor_get_version_info` API, links every LTOIR input from the
manifest, launches the kernel, and checks the returned NVSHMEM version. The
query is self-contained and does not require host-side NVSHMEM initialization.

Run it from the directory against which relative config paths are resolved:

```bash
CARGO_OXIDE=/path/to/cargo-oxide \
CUDA_HOME=/path/to/cuda \
examples/nvshmem_cuda_oxide/run_e2e.sh \
  /path/to/configured/numbast.yaml \
  /tmp/numbast-nvshmem-generated
```

The configured LTOIR and included symbols must come from the same NVSHMEM,
CUDA, macro, and architecture profile used for generation.
The complete profile requires an `sm_100+` CUDA-Oxide target and compatible
hardware for this execution test.

Numbast reads typedefs used by the device API, enum values, and supported
scalar macros from the retained headers. The headers supply team IDs,
comparisons, signal operations, and team configuration masks. Implementation helpers
and the C++-linkage shared-memory convenience functions are excluded from the selected device API.
