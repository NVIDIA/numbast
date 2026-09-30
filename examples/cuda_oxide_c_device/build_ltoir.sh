#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
arch="${1:-sm_80}"
cuda_root="${CUDA_HOME:-/usr/local/cuda}"

cd "${example_dir}"
"${cuda_root}/bin/nvcc" -arch="${arch}" -dc -dlto --keep \
    device_api.cu -o device_api.o

test -s device_api.ltoir
test -s device_api.ptx
awk '$1 == "//" && $2 == ".globl" && $3 ~ /^round1_/ {print $3}' \
    device_api.ptx | sort -u \
    > device_api.symbols
printf '%s\n' "Built ${example_dir}/device_api.ltoir for ${arch}"
