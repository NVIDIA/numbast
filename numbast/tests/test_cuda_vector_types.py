# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from numba import types as nbtypes
from numba.cuda.vector_types import vector_types

from numbast.types import (
    CTYPE_MAPS,
    CUDA_VECTOR_TYPE_MAPS,
    get_numba_type_alignof,
    to_c_type_str,
    to_numba_type,
)


EXPECTED_CUDA_VECTOR_TYPES = (
    ("char1", "int8x1", 1),
    ("char2", "int8x2", 2),
    ("char3", "int8x3", 1),
    ("char4", "int8x4", 4),
    ("uchar1", "uint8x1", 1),
    ("uchar2", "uint8x2", 2),
    ("uchar3", "uint8x3", 1),
    ("uchar4", "uint8x4", 4),
    ("short1", "int16x1", 2),
    ("short2", "int16x2", 4),
    ("short3", "int16x3", 2),
    ("short4", "int16x4", 8),
    ("ushort1", "uint16x1", 2),
    ("ushort2", "uint16x2", 4),
    ("ushort3", "uint16x3", 2),
    ("ushort4", "uint16x4", 8),
    ("int1", "int32x1", 4),
    ("int2", "int32x2", 8),
    ("int3", "int32x3", 4),
    ("int4", "int32x4", 16),
    ("uint1", "uint32x1", 4),
    ("uint2", "uint32x2", 8),
    ("uint3", "uint32x3", 4),
    ("uint4", "uint32x4", 16),
    ("longlong1", "int64x1", 8),
    ("longlong2", "int64x2", 16),
    ("longlong3", "int64x3", 8),
    ("longlong4", "int64x4", 16),
    ("ulonglong1", "uint64x1", 8),
    ("ulonglong2", "uint64x2", 16),
    ("ulonglong3", "uint64x3", 8),
    ("ulonglong4", "uint64x4", 16),
    ("float1", "float32x1", 4),
    ("float2", "float32x2", 8),
    ("float3", "float32x3", 4),
    ("float4", "float32x4", 16),
    ("double1", "float64x1", 8),
    ("double2", "float64x2", 16),
    ("double3", "float64x3", 8),
    ("double4", "float64x4", 16),
)


def test_all_canonical_cuda_vector_types_are_registered():
    expected_cxx_names = {
        cxx_name
        for cxx_name, _numba_key, _alignof in EXPECTED_CUDA_VECTOR_TYPES
    }
    assert set(CUDA_VECTOR_TYPE_MAPS) == expected_cxx_names


@pytest.mark.parametrize(
    "cxx_name,numba_key,alignof", EXPECTED_CUDA_VECTOR_TYPES
)
def test_cuda_vector_types_map_to_numba_types_with_alignment(
    cxx_name,
    numba_key,
    alignof,
):
    result = to_numba_type(cxx_name)

    assert result is vector_types[numba_key]
    assert result is CTYPE_MAPS[cxx_name]
    assert get_numba_type_alignof(result) == alignof
    assert getattr(result, "alignof_", None) is None
    assert CUDA_VECTOR_TYPE_MAPS[cxx_name] == (result, alignof)


@pytest.mark.parametrize(
    "cxx_name,numba_key,_alignof", EXPECTED_CUDA_VECTOR_TYPES
)
def test_cuda_vector_types_map_back_to_c_type_strings(
    cxx_name,
    numba_key,
    _alignof,
):
    assert to_c_type_str(vector_types[numba_key]) == cxx_name


def test_cuda_vector_pointer_types_map_back_to_c_type_strings():
    assert (
        to_c_type_str(nbtypes.CPointer(vector_types["float32x3"])) == "float3*"
    )
