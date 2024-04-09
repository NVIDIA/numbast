#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

yq --version

# Update CUDAToolkit version according to CI environment.
# TODO: we should use some type of yaml template engine to simplify this.
yq e '(.dependencies[] | select(. == "cuda-version*")) |= "cuda-version=${RAPIDS_CUDA_VERSION%.*}"' conda/environment.yaml -i
yq e '(.dependencies[] | select(. == "cuda-python*")) |= "cuda-python=${RAPIDS_CUDA_VERSION%.*}"' conda/environment.yaml -i

rapids-mamba-retry env create --yes -f conda/environment.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
