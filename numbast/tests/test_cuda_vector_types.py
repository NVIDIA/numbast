# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba import types as nbtypes
from numba.cuda.vector_types import vector_types

from numbast.types import to_c_type_str, to_numba_type


def test_cuda_vector_types_map_to_numba_types():
    assert to_numba_type("float3") == vector_types["float32x3"]
    assert to_numba_type("float4") == vector_types["float32x4"]
    assert to_numba_type("uint4") == vector_types["uint32x4"]
    assert to_numba_type("uchar4") == vector_types["uint8x4"]


def test_cuda_vector_types_map_back_to_c_type_strings():
    assert to_c_type_str(vector_types["float32x3"]) == "float3"
    assert to_c_type_str(vector_types["float32x4"]) == "float4"
    assert to_c_type_str(vector_types["uint32x4"]) == "uint4"
    assert to_c_type_str(vector_types["uint8x4"]) == "uchar4"


def test_cuda_vector_pointer_types_map_back_to_c_type_strings():
    assert (
        to_c_type_str(nbtypes.CPointer(vector_types["float32x3"])) == "float3*"
    )
    assert (
        to_c_type_str(nbtypes.CPointer(vector_types["float32x4"])) == "float4*"
    )
