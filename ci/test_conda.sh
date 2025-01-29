#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Starting Conda Package Test"

rapids-logger "Creating Test Environment"
rapids-mamba-retry create -n test \
  click \
  pytest

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-mamba-retry install \
  -c `pwd`/conda-repo \
  ast_canopy \
  numbast \
  numbast-extensions

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
