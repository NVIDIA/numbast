#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild conda/recipes/ast_canopy
rapids-conda-retry mambabuild conda/recipes/numbast
rapids-conda-retry mambabuild conda/recipes/numba_extensions

# run tests
./ci/test_conda.sh

#rapids-upload-conda-to-s3 python
