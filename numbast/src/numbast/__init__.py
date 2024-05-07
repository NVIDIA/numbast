# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numba

# Use pynvjitlink by default. This can avoid version mismatch between system driver and
# installed CTK version.
from pynvjitlink.patch import patch_numba_linker

patch_numba_linker()

from numbast import numba_patch

from numbast.struct import bind_cxx_struct, bind_cxx_structs
from numbast.function import bind_cxx_function, bind_cxx_functions
from numbast.enum import bind_cxx_enum
from numbast.shim_writer import MemoryShimWriter, FileShimWriter

major, minor, *_ = numba.__version__.split(".")
if int(minor) < 59:
    raise RuntimeError("Numba version >= 0.59rc1 is required")
