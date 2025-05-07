# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

__version__ = importlib.metadata.version("ast_canopy")

from ast_canopy.api import (
    parse_declarations_from_source,
    value_from_constexpr_vardecl,
)

__all__ = [
    "__version__",
    "parse_declarations_from_source",
    "value_from_constexpr_vardecl",
]
