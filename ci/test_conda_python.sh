#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Starting Conda Package Test"

rapids-logger "Creating Test Environment"

# Prepend the local conda-repo to the conda config. This is a workaround to
# ensure that the local conda-repo is preferred over the default channels.
# TODO: Add the short sha to the build string and constrain by that.
conda config --set channel_priority strict
conda config --prepend channels `pwd`/conda-repo

# TODO: replace this with rapids-dependency-manager
rapids-mamba-retry create -n test \
  click \
  pytest \
  clangdev >=18 \
  cuda-nvcc >==${RAPIDS_CUDA_VERSION%.*} \
  cuda-version >==${RAPIDS_CUDA_VERSION%.*} \
  cuda-nvrtc \
  numba >=0.59 \
  numba-cuda >=0.2.0 \
  pynvjitlink >=0.2 \
  cuda-cudart-dev \
  python=${RAPIDS_PY_VERSION} \
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
python ci/run_tests.py --ast-canopy --numbast --bf16

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
