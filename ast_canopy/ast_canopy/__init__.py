# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata

__version__ = importlib.metadata.version("ast_canopy")

from ast_canopy.api import (
    parse_declarations_from_source,
    value_from_constexpr_vardecl,
)
from ast_canopy.constants import INVALID_ALIGN_OF, INVALID_SIZE_OF

__all__ = [
    "__version__",
    "INVALID_ALIGN_OF",
    "INVALID_SIZE_OF",
    "parse_declarations_from_source",
    "value_from_constexpr_vardecl",
]
