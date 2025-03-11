# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import os


@pytest.fixture(scope="session")
def data_folder():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    def get_data_file(*file):
        return os.path.join(current_directory, "data/", *file)

    return get_data_file
