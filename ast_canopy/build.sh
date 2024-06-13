#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x -e
set -o pipefail

echo "Beginning astcanopy Build"

env
echo ""

# Relative to ast_canopy/ <-- This is essential for conda build
echo "Making directory..."
mkdir -p ast_canopy/cpp/build
echo "Entering cpp build..."
pushd ast_canopy/cpp/build
echo "Starting cmake config..."
cmake ${CMAKE_ARGS} \
    -GNinja \
    -DCMAKE_BUILD_TYPE:STRING="Release" \
    -DCMAKE_PREFIX_PATH:PATH="${CONDA_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX:PATH="${CONDA_PREFIX}" \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DBUILD_STATIC_LIBS:BOOL=OFF \
    -DCMAKE_CXX_STANDARD:STRING=17 \
    ../
echo "cmake build..."
cmake --build . -j
echo "cmake install..."
cmake --install .
echo "done!"
popd

echo "pip installing..."
python -m pip install ast_canopy/ -vv
