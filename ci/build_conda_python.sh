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

# If there were commits since the tag this is a dev build.
if [[ "${GIT_DESCRIBE_NUMBER}" =~ ^[0-9]+$ ]] && [[ "${GIT_DESCRIBE_NUMBER}" -gt 0 ]]; then
  export PROJECT_VERSION="${PROJECT_VERSION}.dev${GIT_DESCRIBE_NUMBER}"
fi

rapids-print-env

rapids-logger "Begin py build"

sccache --zero-stats

rapids-conda-retry build conda/recipes/ast_canopy
rapids-conda-retry build conda/recipes/numbast
rapids-conda-retry build conda/recipes/numbast_extensions

sccache --show-adv-stats
