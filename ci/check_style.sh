#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

pip install pre-commit

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
