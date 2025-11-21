# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numba

from numbast.struct import bind_cxx_struct, bind_cxx_structs
from numbast.class_template import (
    bind_cxx_class_template_specialization,
    bind_cxx_class_template,
    bind_cxx_class_templates,
)
from numbast.function import bind_cxx_function, bind_cxx_functions
from numbast.enum import bind_cxx_enum, bind_cxx_enums
from numbast.shim_writer import MemoryShimWriter, FileShimWriter

import importlib.metadata

__version__ = importlib.metadata.version("numbast")

major, minor, *_ = numba.__version__.split(".")
if int(minor) < 59:
    raise RuntimeError("Numba version >= 0.59rc1 is required")

__all__ = [
    "__version__",
    "bind_cxx_enum",
    "bind_cxx_enums",
    "bind_cxx_function",
    "bind_cxx_functions",
    "bind_cxx_struct",
    "bind_cxx_structs",
    "bind_cxx_class_template_specialization",
    "bind_cxx_class_template",
    "bind_cxx_class_templates",
    "MemoryShimWriter",
    "FileShimWriter",
]
