#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x -e
set -o pipefail

BUILD_TYPE="Release"
Editable_Mode="false"

usage() {
    echo "Usage: ./build.sh [options]"
    echo ""
    echo "Options:"
    echo "  --develop  Install ast_canopy in editable mode"
    echo "  --debug    Build libastcanopy in debug mode"
    echo "  --help     Show this help message"
    echo ""
    echo "Example usage:"
    echo "  ./build.sh                   # Build and install in release mode"
    echo "  ./build.sh --develop         # Build and install in editable mode"
    echo "  ./build.sh --debug           # Build and install in debug mode"
    echo "  ./build.sh --develop --debug # Build and install in editable and debug mode"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --develop)
            Editable_Mode="true"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "Beginning astcanopy Build"

# Clean the build cache
echo "Cleaning ast_canopy/cpp/build cache..."
rm -rf ast_canopy/cpp/build
mkdir -p ast_canopy/cpp/build
echo "Cache cleaned. Starting fresh build..."

env
echo ""

# Relative to ast_canopy/ <-- This is essential for conda build
echo "Entering cpp build..."
pushd ast_canopy/cpp/build
echo "Starting cmake config..."
# CMake automatically reads CMAKE_PREFIX_PATH and CMAKE_INSTALL_PREFIX from environment variables
cmake ${CMAKE_ARGS} \
    -GNinja \
    -DCMAKE_BUILD_TYPE:STRING="${BUILD_TYPE}" \
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

if [ "$Editable_Mode" = "true" ]; then
    # If it's set, perform an editable install of ast_canopy
    echo "pip installing in editable mode..."
    pip install -e ast_canopy/ -vv
else
    # If not, perform a normal install
    echo "pip installing..."
    python -m pip install ast_canopy/ -vv
fi
