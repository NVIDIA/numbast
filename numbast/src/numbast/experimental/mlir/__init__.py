# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numbast.experimental.mlir.struct import bind_cxx_struct, bind_cxx_structs
from numbast.experimental.mlir.class_template import (
    bind_cxx_class_template_specialization,
    bind_cxx_class_template,
    bind_cxx_class_templates,
    clear_concrete_type_caches,
)
from numbast.experimental.mlir.function import (
    bind_cxx_function,
    bind_cxx_functions,
)
from numbast.experimental.mlir.function_template import (
    bind_cxx_function_template,
    bind_cxx_function_templates,
)
from numbast.experimental.mlir.enum import bind_cxx_enum, bind_cxx_enums
from numbast.experimental.mlir.shim_writer import (
    MemoryShimWriter,
    FileShimWriter,
)

import importlib.metadata

try:
    __version__ = importlib.metadata.version("numbast")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "bind_cxx_enum",
    "bind_cxx_enums",
    "bind_cxx_function",
    "bind_cxx_functions",
    "bind_cxx_function_template",
    "bind_cxx_function_templates",
    "bind_cxx_struct",
    "bind_cxx_structs",
    "bind_cxx_class_template_specialization",
    "bind_cxx_class_template",
    "bind_cxx_class_templates",
    "clear_concrete_type_caches",
    "MemoryShimWriter",
    "FileShimWriter",
]
