#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

CUDATOOLKIT_VERSION=$1
PYTHON_VERSION=$2

rapids-logger "Running Numbast Setup, Build and Tests."

echo "CUDATOOLKIT_VERSION=${CUDATOOLKIT_VERSION}"
echo "PYTHON_VERSION=${PYTHON_VERSION}"

CONDA_RECIPE_TEMPLATE="conda/environment_template.yaml"
CONDA_RECIPE="conda/environment.yaml"

# Install environments
python ci/instantiate_dep_yaml.py $CONDA_RECIPE_TEMPLATE $CONDA_RECIPE $CUDATOOLKIT_VERSION $PYTHON_VERSION
rapids-print-env
rapids-mamba-retry env create --yes -f $CONDA_RECIPE -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

which nvcc
nvcc --version

rapids-logger "Install Numbast and all submodules"

# Install AST_Canopy, Numbast and extensions
ast_canopy/build.sh
pip install numbast/
# bf16 is now in numbast_extensions.bf16
pip install numbast_extensions/
pip install numbast_extensions/fp16
pip install numbast_extensions/curand_device
pip install numbast_extensions/curand_host

ci/run_tests.sh
