# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for ``numbast.types.register_cxx_type``.

``register_cxx_type`` lets callers register a C++ type name in numbast's
type registry without parsing it from a header. After registration,
``to_numba_type(cxx_name)`` resolves to the registered numba type instead
of ``Opaque``. This is how external types (e.g. ``Eigen::half``) get
mapped to ``float16`` when their real headers are too complex to parse.
"""

from numba import types as nbtypes

from numbast.types import register_cxx_type, to_numba_type


def test_unregistered_type_is_opaque():
    """Before registration, to_numba_type returns an Opaque type."""
    result = to_numba_type("SomeLibrary::Unregistered_abc123")
    assert isinstance(result, nbtypes.Opaque)


def test_register_maps_to_numba_scalar():
    """Registering a C++ name routes it to the given numba scalar type."""
    cxx_name = "SomeLibrary::MyHalf_abc123"
    register_cxx_type(cxx_name, nbtypes.float16)

    result = to_numba_type(cxx_name)
    assert result is nbtypes.float16


def test_register_overwrites_previous_registration():
    """Re-registering the same name swaps the mapping."""
    cxx_name = "SomeLibrary::Swappable_abc123"
    register_cxx_type(cxx_name, nbtypes.int32)
    assert to_numba_type(cxx_name) is nbtypes.int32

    register_cxx_type(cxx_name, nbtypes.float64)
    assert to_numba_type(cxx_name) is nbtypes.float64
