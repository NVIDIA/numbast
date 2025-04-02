# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources

__version__ = (
    importlib.resources.files(__package__).joinpath("VERSION").read_text().strip()
)

__all__ = ["__version__"]
