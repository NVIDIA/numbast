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

usage() {
    echo "Usage: ./build.sh [options]"
    echo ""
    echo "Options:"
    echo "  --develop  Install ast_canopy in editable mode"
    echo "  --debug    Build libastcanopy in debug mode"
    echo "  --help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ASTCANOPY_INSTALL_PATH  Override the install path for libastcanopy"
    echo "                          (default: CMAKE_INSTALL_PREFIX)"
    echo ""
    echo "Example usage:"
    echo "  ./build.sh                                    # Build and install in release mode"
    echo "  ./build.sh --develop                          # Build and install in editable mode"
    echo "  ./build.sh --debug                            # Build and install in debug mode"
    echo "  ./build.sh --develop --debug                  # Build and install in editable and debug mode"
    echo "  ASTCANOPY_INSTALL_PATH=/custom/path ./build.sh # Install libastcanopy to custom path"
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

# Set CMAKE paths based on ASTCANOPY_INSTALL_PATH or conda environment detection
if [ -n "$ASTCANOPY_INSTALL_PATH" ]; then
    echo "ASTCANOPY_INSTALL_PATH is set to: $ASTCANOPY_INSTALL_PATH. Using custom install path."
    export CMAKE_PREFIX_PATH="$ASTCANOPY_INSTALL_PATH"
    export CMAKE_INSTALL_PREFIX="$ASTCANOPY_INSTALL_PATH"
else
    echo "Detecting conda environment..."
    IS_CONDA=$($PYTHON_EXECUTABLE "${SCRIPT_DIR}/detect_conda.py")
    if [ "$IS_CONDA" = "true" ]; then
        echo "Conda environment detected. Setting CMAKE_PREFIX_PATH and CMAKE_INSTALL_PREFIX to CONDA_PREFIX: $CONDA_PREFIX"
        export CMAKE_PREFIX_PATH="$CONDA_PREFIX"
        export CMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
    else
        echo "Not in conda environment. Using default CMAKE paths."
    fi
fi

# Relative to ast_canopy/ <-- This is essential for conda build
echo "Entering cpp build..."
pushd "${SCRIPT_DIR}/cpp/build"
echo "Starting cmake config..."
# CMake automatically reads CMAKE_PREFIX_PATH and CMAKE_INSTALL_PREFIX from environment variables
cmake ${CMAKE_ARGS} \
    -G"${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE:STRING="${BUILD_TYPE}" \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DBUILD_STATIC_LIBS:BOOL=OFF \
    -DCMAKE_CXX_STANDARD:STRING=17 \
    -DCMAKE_CXX_FLAGS:STRING="-frtti" \
    -DCMAKE_PREFIX_PATH:STRING="$CMAKE_PREFIX_PATH" \
    -DCMAKE_INSTALL_PREFIX:STRING="$CMAKE_INSTALL_PREFIX" \
    -DLLVM_ENABLE_RTTI:BOOL=ON \
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
