# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast_extensions.bf16.bf16_bindings import *  # noqa: F403
from numbast_extensions.bf16.torch_interop import patch_numba

patch_numba()
