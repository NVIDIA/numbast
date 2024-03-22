# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np


@pytest.fixture(
    params=[
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        # np.float16, # Bug with ptx argument type, b32 instead of b16?
        np.float32,
        np.float64,
    ]
)
def dtype(request):
    return request.param
