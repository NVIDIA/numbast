# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ast_canopy.api import (
    parse_declarations_from_source,
)

from ast_canopy._version import __version__

__all__ = [
    "__version__",
    "parse_declarations_from_source",
]
