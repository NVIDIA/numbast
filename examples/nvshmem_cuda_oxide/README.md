# NVSHMEM Round 1 profile

This directory contains the library-specific configuration needed to replace
NVSHMEM4Rust's handwritten device renderer with Numbast's CUDA-Oxide backend.
The Numbast generator owns declaration parsing, C ABI normalization, Rust name
handling, extern rendering, and the link manifest. NVSHMEM retains ownership
of its header profile, constants that exist only in the preprocessor, LTOIR,
and symbol inventory.

Configure the placeholders in `numbast.yaml.in` from the same NVSHMEM and CUDA
build used to create the device LTOIR, then run:

```bash
numbast-cuda-oxide \
  --cfg-path /path/to/configured/numbast.yaml \
  --output-dir /path/to/nvshmem4rust/runtime/src
```

The five retained headers form the selected declaration profile. Numbast
derives `__CUDA_ARCH__` from `GPU Arch`, which is required because these
headers guard the device API with that macro. `nvshmemi_` implementation
symbols, the internal `nvshmemi_signal` template, and CUDA's
`atomicAdd_system` helper are intentionally outside the public C API.

The symbol inventory is the exact selected public API surface, not an
unfiltered object dump. Retain PTX alongside the LTOIR during the NVSHMEM NVCC
compilation, then extract the public device symbols from that compiler output:

```bash
awk '$1 == "//" && $2 == ".globl" && $3 ~ /^nvshmemx?_/' \
  nvshmem_device.ptx \
  | awk '{print $3}' \
  | sort -u > nvshmem_device_api.symbols
```

Generation fails for symbols missing from either side of that comparison. On
NVSHMEM commit `b0d9d3`, the `sm_80` profile was exercised with 2,730 generated
public C device functions, 50 `nvshmemi_` exclusions, one configured
CUDA-helper exclusion, and one template exclusion. The count is a validation
snapshot, not a hardcoded generator rule; header changes are surfaced by
symbol-parity failure.

No C or C++ shim is generated. The resulting Rust module calls native
NVSHMEM C symbols directly, and CUDA-Oxide links the supplied device LTOIR.

NVSHMEM includes `char`, `short`, half, and bfloat16 APIs with sub-32-bit
values passed by value. CUDA-Oxide supports those exact extern ABIs only on its
modern `sm_100+` NVVM path. The generated manifest lists the affected symbols
and reports whether the selected architecture supports the entire module.
Pointer-based variants and the remaining APIs continue to work on older
targets; Numbast does not silently widen or shim the incompatible ABI.

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

The configured LTOIR and symbol inventory must come from the same NVSHMEM,
CUDA, macro, and architecture profile used for generation.
