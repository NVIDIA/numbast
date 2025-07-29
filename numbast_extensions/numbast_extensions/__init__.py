# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numbast_extensions.bf16

import importlib.metadata

__version__ = importlib.metadata.version("numbast_extensions")

warnings.warn(
    DeprecationWarning(
        "Bfloat16 bindings from numbast_extensions are deprecated. Please use Numba-CUDA's built-in bfloat16 support."
    )
)
