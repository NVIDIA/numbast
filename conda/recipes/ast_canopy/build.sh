#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x -e
set -o pipefail

echo ""

env

echo ""

BUILD_DIR=$SRC_DIR/cpp/build
CPP_DIR=$SRC_DIR/cpp

echo SRC_DIR=$SRC_DIR
echo CPP_DIR=$CPP_DIR
echo BUILD_DIR=$BUILD_DIR

# Relative to ast_canopy/ <-- This is essential for conda build
echo "Making directory..."
mkdir -p cpp/build
echo "entering cpp build..."
cd $BUILD_DIR
echo "starting cmake config..."
cmake $CPP_DIR
echo "cmake build..."
cmake --build $BUILD_DIR -j
echo "cmake install..."
cmake --install $BUILD_DIR
echo "done!"
cd $SRC_DIR

echo "pip installing..."
$PYTHON -m pip install $SRC_DIR -vv
