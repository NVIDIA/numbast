#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
example_dir="${repo_root}/examples/cuda_oxide_c_device"
arch="${1:-sm_80}"
cargo_oxide="${CARGO_OXIDE:-cargo-oxide}"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"
python_executable="${PYTHON:-python}"

# CUDA's header discovery gives CUDA_PATH priority over CUDA_HOME. Keep AST
# parsing, NVCC LTOIR generation, and device linking on one toolkit profile.
export CUDA_HOME="${cuda_root}"
export CUDA_PATH="${cuda_root}"

cd "${repo_root}"
configured_arch="$("${python_executable}" -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["GPU Arch"][0])' \
    "${example_dir}/numbast.yaml")"
if [[ "${arch}" != "${configured_arch}" ]]; then
    echo "requested ${arch}, but numbast.yaml selects ${configured_arch}" >&2
    echo "update GPU Arch in numbast.yaml so parsing, LTOIR, and Rust agree" >&2
    exit 2
fi

"${example_dir}/build_ltoir.sh" "${arch}"

"${python_executable}" -m numbast.tools.cuda_oxide_binding_generator \
    --cfg-path "examples/cuda_oxide_c_device/numbast.yaml" \
    --output-dir "examples/cuda_oxide_c_device"

cd "${example_dir}"
"${cargo_oxide}" run --emit-nvvm-ir --arch="${arch}"
