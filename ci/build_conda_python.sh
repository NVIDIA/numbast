#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

# TODO: migrate to rattler-build, install boa for now to get the build working.
rapids-conda-retry install boa

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild conda/recipes/ast_canopy
rapids-conda-retry mambabuild conda/recipes/numbast
rapids-conda-retry mambabuild conda/recipes/numbast_extensions
