#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild conda/recipes/ast_canopy
rapids-conda-retry mambabuild conda/recipes/numbast
rapids-conda-retry mambabuild conda/recipes/bf16

# run tests
pip install pytest
./ci/run_tests.sh

#rapids-upload-conda-to-s3 python
