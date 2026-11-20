#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# rstcheck compiles C++ documentation snippets with g++.
if ! command -v g++ >/dev/null 2>&1; then
    apt-get update
    apt-get install --yes --no-install-recommends g++
fi

pip install pre-commit

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
