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
GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)

if [[ "${GIT_DESCRIBE_NUMBER}" =~ ^[0-9]+$ ]] && [[ "${GIT_DESCRIBE_NUMBER}" -gt 0 ]]; then
  PROJECT_VERSION="${PROJECT_VERSION}.dev${GIT_DESCRIBE_NUMBER}"
fi

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
  pip
  cffi
  ast_canopy="${PROJECT_VERSION}"
  numbast="${PROJECT_VERSION}"
)

if [[ "${TEST_BACKEND}" == "numba-cuda" ]]; then
  TEST_ENV_PACKAGES+=(
    "numba-cuda>=0.25.0"
  )
fi

rapids-mamba-retry create -n test "${TEST_ENV_PACKAGES[@]}"

if [[ "${TEST_BACKEND}" == "mlir" ]]; then
  rapids-logger "Removing numba-cuda from MLIR test environment"
  if conda list -n test numba-cuda | grep -E '^numba-cuda[[:space:]]'; then
    conda remove -n test --force -y numba-cuda
  fi
fi

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

if [[ "${TEST_BACKEND}" == "mlir" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  rapids-logger "Installing numba-cuda-mlir with pip"
  python -m pip install numba-cuda-mlir
  if python -m pip show numba-cuda; then
    python -m pip uninstall -y numba-cuda
  fi
fi

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
  python ci/run_tests.py --ast-canopy --numbast
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
