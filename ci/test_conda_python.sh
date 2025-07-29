#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Starting Conda Package Test"

rapids-logger "Creating Test Environment"
# TODO: replace this with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
  -c `pwd`/conda-repo \
  click \
  pytest \
  clangdev >=18 \
  cuda-nvcc \
  cuda-version=${RAPIDS_CUDA_VERSION%.*} \
  cuda-nvrtc \
  numba >=0.59 \
  numba-cuda >=0.2.0 \
  pynvjitlink >=0.2 \
  cuda-cudart-dev \
  python=${RAPIDS_PY_VERSION} \
  cffi \
  ast_canopy \
  numbast \
  numbast-extensions

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
# Debug print
python ci/run_tests.py --ast-canopy --numbast

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
