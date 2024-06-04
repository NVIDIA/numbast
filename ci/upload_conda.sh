#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CONDA_TOKEN="is-309f6f5c-e5ce-49aa-b756-244bb5c45beb"

PKGS_TO_UPLOAD=$(rapids-find-anaconda-uploads.py $RAPIDS_CONDA_BLD_OUTPUT_DIR)

echo $PKGS_TO_UPLOAD

rapids-retry anaconda \
    -t $CONDA_TOKEN \
    upload \
    --skip-existing \
    --no-progress \
    ${PKGS_TO_UPLOAD}
