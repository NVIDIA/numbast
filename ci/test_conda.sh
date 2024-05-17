#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Creating Test Environment"
rapids-mamba-retry create -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-mamba-retry install \
  ast_canopy \
  numbast \
  numba-extensions-bf16 \
  --channel $RAPIDS_CONDA_BLD_OUTPUT_DIR/linux-64

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
./ci/run_tests.sh

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
