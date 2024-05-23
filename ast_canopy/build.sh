#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SP_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo "Site package path: $SP_PATH"

echo "Installing ast_canopy..."
pip install ast_canopy/

echo "Moving $SP_PATH/libastcanopy.so to $CONDA_PREFIX/lib"
mv $SP_PATH/libastcanopy.so $CONDA_PREFIX/lib
