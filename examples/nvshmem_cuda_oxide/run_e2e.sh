#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 CONFIG_YAML OUTPUT_DIRECTORY" >&2
    exit 2
fi

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="$(realpath "$1")"
output_dir="$(realpath -m "$2")"
config_base="$(pwd)"
python_executable="${PYTHON:-python}"
cargo_oxide="${CARGO_OXIDE:-cargo-oxide}"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"

export CUDA_HOME="${cuda_root}"
export CUDA_PATH="${cuda_root}"

mkdir -p "${output_dir}"
"${python_executable}" -m numbast.tools.cuda_oxide_binding_generator \
    --cfg-path "${config_path}" \
    --output-dir "${output_dir}"

export NUMBAST_CONFIG_BASE="${config_base}"
export NUMBAST_NVSHMEM_RUST_BINDINGS="${output_dir}/nvshmem_device.rs"
export NUMBAST_NVSHMEM_MANIFEST="${output_dir}/nvshmem_device.manifest.json"

arch="$(${python_executable} -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["target"]["gpu_arch"])' \
    "${NUMBAST_NVSHMEM_MANIFEST}")"

cd "${example_dir}/e2e"
"${cargo_oxide}" run --emit-nvvm-ir --arch="${arch}"
