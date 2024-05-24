#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

mkdir -p ast_canopy/cpp/build
pushd ast_canopy/cpp/build
cmake ../
cmake build .
cmake install .
popd

pip install ast_canopy/ -vv
