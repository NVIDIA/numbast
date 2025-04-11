#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

# TODO: Once tagging is in place we can use normal git describe.
# This counts the number of commits since the first commit.
export PROJECT_VERSION=$(<VERSION)
export GIT_DESCRIBE_NUMBER=$(git rev-list 5f486a60..HEAD --count)
export GIT_DESCRIBE_HASH=$(git rev-parse --short HEAD)

rapids-print-env

rapids-logger "Begin py build"

sccache --zero-stats

rapids-conda-retry build conda/recipes/ast_canopy
rapids-conda-retry build conda/recipes/numbast
rapids-conda-retry build conda/recipes/numbast_extensions

sccache --show-adv-stats
