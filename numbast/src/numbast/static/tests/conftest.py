# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import os


@pytest.fixture
def sample_struct():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(current_directory, "data/", "sample_struct_static_bindings.cuh")
