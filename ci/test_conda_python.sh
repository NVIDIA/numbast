#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Starting Conda Package Test"

GIT_DESCRIBE_TAG=$(git describe --abbrev=0)
PROJECT_VERSION=${GIT_DESCRIBE_TAG:1}
GIT_DESCRIBE_HASH=$(git rev-parse --short HEAD)

rapids-logger "Creating Test Environment"
# TODO: replace this with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
  -c `pwd`/conda-repo \
  click \
  pytest \
  "clangdev>=18,<22.0" \
  cuda-nvcc \
  cuda-version=${RAPIDS_CUDA_VERSION%.*} \
  cuda-nvrtc \
  numba >=0.59 \
  "numba-cuda>=0.21.0,<0.28.0" \
  cuda-cudart-dev \
  python=${RAPIDS_PY_VERSION} \
  cffi \
  ast_canopy="${PROJECT_VERSION}=*g${GIT_DESCRIBE_HASH}*" \
  numbast="${PROJECT_VERSION}=*g${GIT_DESCRIBE_HASH}*" \
  numbast-extensions="${PROJECT_VERSION}=*g${GIT_DESCRIBE_HASH}*"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
python ci/run_tests.py --ast-canopy --numbast --cccl

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
