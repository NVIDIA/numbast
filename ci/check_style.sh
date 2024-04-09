#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

# Update CUDAToolkit version according to CI environment.
# TODO: we should use some type of yaml template engine to simplify this.
yq -i -y '.dependencies |= map(if startswith("cuda-version") then "cuda-version=${RAPIDS_CUDA_VERSION%.*}" else . end)' conda/environment.yaml
yq -i -y '.dependencies |= map(if startswith("cuda-python") then "cuda-python=${RAPIDS_CUDA_VERSION%.*}" else . end)' conda/environment.yaml

rapids-mamba-retry env create --yes -f conda/environment.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
