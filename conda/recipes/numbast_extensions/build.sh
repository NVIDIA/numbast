#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x -e
set -o pipefail

echo ""

env

echo ""

cd numbast_extensions
$PYTHON -m pip install . -vv
