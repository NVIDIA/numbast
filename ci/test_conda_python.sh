#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Starting Conda Package Test"

TEST_BACKEND=${TEST_BACKEND:-numba-cuda}
if [[ "${TEST_BACKEND}" != "numba-cuda" ]] && [[ "${TEST_BACKEND}" != "mlir" ]]; then
  echo "Unsupported TEST_BACKEND='${TEST_BACKEND}'. Expected 'numba-cuda' or 'mlir'."
  exit 1
fi

GIT_DESCRIBE_TAG=$(git describe --abbrev=0)
PROJECT_VERSION=${GIT_DESCRIBE_TAG:1}
GIT_DESCRIBE_HASH=$(git rev-parse --short HEAD)

rapids-logger "Creating Test Environment"
# TODO: replace this with rapids-dependency-file-generator
TEST_ENV_PACKAGES=(
  -c "$(pwd)/conda-repo"
  click
  pytest
  "clangdev>=18,<22.0"
  cuda-nvcc
  cuda-version=${RAPIDS_CUDA_VERSION%.*}
  cuda-nvrtc
  cuda-cudart-dev
  python=${RAPIDS_PY_VERSION}
  cffi
  ast_canopy="${PROJECT_VERSION}=*g${GIT_DESCRIBE_HASH}*"
  numbast="${PROJECT_VERSION}=*g${GIT_DESCRIBE_HASH}*"
)

if [[ "${TEST_BACKEND}" == "mlir" ]]; then
  TEST_ENV_PACKAGES+=(
    numba-cuda-mlir
  )
else
  TEST_ENV_PACKAGES+=(
    "numba-cuda>=0.25.0"
    numbast-extensions="${PROJECT_VERSION}=*g${GIT_DESCRIBE_HASH}*"
  )
fi

rapids-mamba-retry create -n test "${TEST_ENV_PACKAGES[@]}"

if [[ "${TEST_BACKEND}" == "mlir" ]]; then
  rapids-logger "Removing numba-cuda from MLIR test environment"
  conda remove -n test --force -y numba-cuda
fi

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

if [[ "${TEST_BACKEND}" == "mlir" ]]; then
  rapids-logger "Check MLIR test dependencies"
  python -c "import numba_cuda_mlir"
  conda list numba-cuda | grep -E '^numba-cuda[[:space:]]' && exit 1 || true
else
  rapids-logger "Show Numba system info"
  python -m numba --sysinfo
fi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
if [[ "${TEST_BACKEND}" == "mlir" ]]; then
  python ci/run_tests.py --ast-canopy --mlir
else
  python ci/run_tests.py --ast-canopy --numbast --cccl
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
