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

# Convert a colon-separated path list (env style) into a
# semicolon-separated list (CMake style)
normalize_cmake_prefix_path() {
    local input="$1"
    if [[ -z "$input" ]]; then
        echo ""
    else
        echo "${input//:/;}"
    fi
}

usage() {
    echo "Usage: ./build.sh [options]"
    echo ""
    echo "Options:"
    echo "  --develop           Install ast_canopy in editable mode"
    echo "  --debug             Build libastcanopy in debug mode"
    echo "  --llvm-linkage=TYPE Set LLVM linkage type (STATIC or SHARED, default: SHARED)"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  ASTCANOPY_INSTALL_PATH  Install prefix for libastcanopy (sets CMAKE_INSTALL_PREFIX)"
    echo "                          Also appended to CMAKE_PREFIX_PATH when provided"
    echo "  CMAKE_PREFIX_PATH       Semicolon-separated list of prefixes for CMake package discovery"
    echo "                          Can be used together with ASTCANOPY_INSTALL_PATH"
    echo ""
    echo "Example usage:"
    echo "  ./build.sh                                    # Build and install in release mode with shared LLVM"
    echo "  ./build.sh --develop                          # Build and install in editable mode"
    echo "  ./build.sh --debug                            # Build and install in debug mode"
    echo "  ./build.sh --llvm-linkage=STATIC              # Build with static LLVM linking"
    echo "  ./build.sh --develop --debug                  # Build and install in editable and debug mode"
    echo "  ./build.sh --llvm-linkage=STATIC --debug      # Build with static LLVM and debug mode"
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

# Determine base CMAKE_PREFIX_PATH and default CMAKE_INSTALL_PREFIX
# FIXME: We currently rely on environment CMAKE_PREFIX_PATH, which is often
#        colon-delimited like PATH. CMake expects semicolons. We only
#        normalize when passing it as a CMake option below; revisit this
#        behavior to avoid delimiter confusion in the environment.
if [ -n "$CMAKE_PREFIX_PATH" ]; then
    echo "Using CMAKE_PREFIX_PATH from environment: $CMAKE_PREFIX_PATH"
fi

if [ -n "$CMAKE_INSTALL_PREFIX" ]; then
    echo "Using CMAKE_INSTALL_PREFIX from environment: $CMAKE_INSTALL_PREFIX"
else
    IS_CONDA=$($PYTHON_EXECUTABLE "${SCRIPT_DIR}/detect_conda.py")
    if [ "$IS_CONDA" = "true" ]; then
        echo "Conda environment detected. Setting install path to CONDA_PREFIX: $CONDA_PREFIX"
        export CMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
    else
        echo "Not in conda environment. Leaving CMAKE_INSTALL_PREFIX as default unless provided."
    fi
fi

# Apply ASTCANOPY_INSTALL_PATH independently
if [ -n "$ASTCANOPY_INSTALL_PATH" ]; then
    echo "ASTCANOPY_INSTALL_PATH is set to: $ASTCANOPY_INSTALL_PATH"
    export CMAKE_INSTALL_PREFIX="$ASTCANOPY_INSTALL_PATH"
    if [ -n "$CMAKE_PREFIX_PATH" ]; then
        # FIXME: Appending with ';' here mixes CMake-style delimiter into the
        #        environment variable. Consider standardizing handling.
        export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH};${ASTCANOPY_INSTALL_PATH}"
    else
        export CMAKE_PREFIX_PATH="${ASTCANOPY_INSTALL_PATH}"
    fi
fi

echo "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
echo "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}"

echo "Entering cpp build..."
pushd "${SCRIPT_DIR}/cpp/build"
echo "Starting cmake config..."
echo "Build configuration:"
echo "  BUILD_TYPE: ${BUILD_TYPE}"
echo "  LLVM_LINKAGE: ${LLVM_LINKAGE}"
echo "  Editable_Mode: ${Editable_Mode}"
echo ""
# CMake automatically reads CMAKE_PREFIX_PATH and CMAKE_INSTALL_PREFIX from environment variables
cmake ${CMAKE_ARGS} \
    -G"${CMAKE_GENERATOR}" \
    -DCMAKE_BUILD_TYPE:STRING="${BUILD_TYPE}" \
    -DBUILD_SHARED_LIBS:BOOL=ON \
    -DCMAKE_CXX_STANDARD:STRING=17 \
    -DCMAKE_CXX_FLAGS:STRING="-frtti" \
    -DCMAKE_PREFIX_PATH:STRING="$(normalize_cmake_prefix_path "$CMAKE_PREFIX_PATH")" \
    -DCMAKE_INSTALL_PREFIX:STRING="$CMAKE_INSTALL_PREFIX" \
    -DLLVM_ENABLE_RTTI:BOOL=ON \
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
