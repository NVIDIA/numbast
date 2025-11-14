#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

# Generate package version from git tags.
GIT_DESCRIBE_TAG=$(git describe --abbrev=0)
export PROJECT_VERSION=${GIT_DESCRIBE_TAG:1}
export GIT_DESCRIBE_NUMBER=$(git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count)
export GIT_DESCRIBE_HASH=$(git rev-parse --short HEAD)

rapids-print-env

rapids-logger "Begin py build"

sccache --zero-stats

rapids-conda-retry build conda/recipes/ast_canopy
rapids-conda-retry build conda/recipes/numbast
rapids-conda-retry build conda/recipes/numbast_extensions

sccache --show-adv-stats
