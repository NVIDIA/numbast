#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "source directory: " $SRC_DIR

# Relative to ast_canopy/ <-- This is essential for conda build
mkdir -p cpp/build
pushd cpp/build
cmake ../
cmake --build . -j
cmake --install .
popd

pip install ast_canopy/ -vv
