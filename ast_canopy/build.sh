#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x -e
set -o pipefail

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python}"

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

BUILD_TYPE="Release"
Editable_Mode="false"
LLVM_LINKAGE="SHARED"

usage() {
    echo "Usage: ./build.sh [options]"
    echo ""
    echo "Options:"
    echo "  --develop              Install ast_canopy in editable mode"
    echo "  --debug                Build libastcanopy in debug mode"
    echo "  --llvm-linkage=TYPE    Set LLVM linkage type (STATIC or SHARED, default: SHARED)"
    echo "  --help                 Show this help message"
    echo ""
    echo "Example usage:"
    echo "  ./build.sh                           # Build and install in release mode with shared LLVM"
    echo "  ./build.sh --develop                 # Build and install in editable mode"
    echo "  ./build.sh --debug                   # Build and install in debug mode"
    echo "  ./build.sh --llvm-linkage=STATIC     # Build with static LLVM linking"
    echo "  ./build.sh --develop --debug         # Build and install in editable and debug mode"
    echo "  ./build.sh --llvm-linkage=STATIC --debug # Build with static LLVM and debug mode"
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
        --llvm-linkage=*)
            LLVM_LINKAGE="${1#*=}"
            if [[ "$LLVM_LINKAGE" != "STATIC" && "$LLVM_LINKAGE" != "SHARED" ]]; then
                echo "Error: --llvm-linkage must be either STATIC or SHARED, got: $LLVM_LINKAGE"
                usage
                exit 1
            fi
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

# Detect build system generator
if command -v ninja >/dev/null 2>&1; then
    CMAKE_GENERATOR="Ninja"
    echo "Ninja detected. Using Ninja build system."
else
    CMAKE_GENERATOR="Unix Makefiles"
    echo "Ninja not found. Falling back to Unix Makefiles."
fi

# Clean the build cache
echo "Cleaning ast_canopy/cpp/build cache..."
rm -rf "${SCRIPT_DIR}/cpp/build"
mkdir -p "${SCRIPT_DIR}/cpp/build"
echo "Cache cleaned. Starting fresh build..."

env
echo ""

# Relative to ast_canopy/ <-- This is essential for conda build
echo "Entering cpp build..."
pushd "${SCRIPT_DIR}/cpp/build"
echo "Starting cmake config..."
echo "Build configuration:"
echo "  BUILD_TYPE: ${BUILD_TYPE}"
echo "  LLVM_LINKAGE: ${LLVM_LINKAGE}"
echo "  Editable_Mode: ${Editable_Mode}"
echo ""

cmake ${CMAKE_ARGS} \
    -G"${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE:STRING="${BUILD_TYPE}" \
    -DCMAKE_PREFIX_PATH:PATH="${CONDA_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX:PATH="${CONDA_PREFIX}" \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DBUILD_STATIC_LIBS:BOOL=OFF \
    -DCMAKE_CXX_STANDARD:STRING=17 \
    -DLLVM_LINKAGE:STRING="${LLVM_LINKAGE}" \
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
    $PYTHON_EXECUTABLE -m pip install -e "${SCRIPT_DIR}/" -vv
else
    # If not, perform a normal install
    echo "pip installing..."
    $PYTHON_EXECUTABLE -m pip install "${SCRIPT_DIR}/" -vv
fi
