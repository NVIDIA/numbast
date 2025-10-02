#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

exit 0

LLVM_PREFIX=${LLVM_PREFIX:-${GITHUB_WORKSPACE}/llvm-install}

echo "=========================================="
echo "Building project with LLVM"
echo "LLVM prefix: ${LLVM_PREFIX}"
echo "=========================================="

echo "=========================================="
echo "Building ast_canopy with built LLVM"
echo "=========================================="

# Set environment variables for ast_canopy build
export CMAKE_PREFIX_PATH="$LLVM_PREFIX:${CMAKE_PREFIX_PATH:-}"
export PATH="${LLVM_PREFIX}/bin:$PATH"

# Build ast_canopy
cd "$GITHUB_WORKSPACE/ast_canopy"
echo "Building ast_canopy..."
./build.sh --llvm-linkage=STATIC

echo "Build completed successfully!"
