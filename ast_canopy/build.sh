#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SP_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
echo "Site package path: $SP_PATH"

echo "Installing ast_canopy..."
pip install ast_canopy/

# Temporarily allow unbound variables for conda env detection.
set +u
if [[ -n "$CONDA_PREFIX" ]]; then
    echo "In conda environment, moving $SP_PATH/libastcanopy.so to $CONDA_PREFIX/lib"
    mv $SP_PATH/libastcanopy.so $CONDA_PREFIX/lib
else
    echo "Cannot detect conda environment, installing $SP_PATH/libastcanopy.so to system lib."
    echo "Assuming /usr/local/lib as the default system library path"
    mv $SP_PATH/libastcanopy.so /usr/local/lib/
    ldconfig
fi
set -u
