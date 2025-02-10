# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest


@pytest.fixture(scope="session")
def data_folder():
    current_directory = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    return current_directory / "data"
