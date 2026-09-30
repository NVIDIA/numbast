# Direct C device binding with CUDA-Oxide

This is the Round 1 end-to-end artifact. Numbast parses a CUDA header, emits
`bindings.rs` and `bindings.manifest.json`, CUDA compiles the implementation as
LTOIR, and a CUDA-Oxide kernel calls the generated `add` alias. There is no
generated or handwritten C++ shim.

From the Numbast repository root, with Numbast installed from this checkout and
CUDA-Oxide's `cargo oxide` command installed:

```bash
examples/cuda_oxide_c_device/run.sh sm_80
```

Choose an architecture supported by the installed toolkit and GPU. The
architecture passed to `run.sh` must also match `GPU Arch` in `numbast.yaml`;
edit that value together when using something other than `sm_80`.

The generated manifest is the handoff to the CUDA-Oxide/link step. It records
the architecture, expected LTOIR input, native and prefix-stripped Rust symbol
names, normalized C/Rust types, excluded declarations, source hashes, and tool
versions.

`build_ltoir.sh` also refreshes `device_api.symbols` from NVCC's retained PTX
symbol declarations for the same LTOIR compilation. Generation checks exact
parity with that device-symbol inventory before writing either output artifact.
The Rust executable reads the manifest, asks
CUDA-Oxide's artifact finalizer to link its kernel LTOIR with every ordered
`LTOIR Inputs` entry, loads the resulting cubin, and verifies the result on the
GPU.
