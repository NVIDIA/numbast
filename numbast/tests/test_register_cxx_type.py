# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for ``numbast.types.register_cxx_type``.

``register_cxx_type`` lets callers register a C++ type name in numbast's
type registry without parsing it from a header. After registration,
``to_numba_type(cxx_name)`` resolves to the registered numba type instead
of ``Opaque``. This is how external types (e.g. ``Eigen::half``) get
mapped to ``float16`` when their real headers are too complex to parse.
"""

import pytest
from numba import types as nbtypes

from numbast.types import CTYPE_MAPS, register_cxx_type, to_numba_type


@pytest.fixture(autouse=True)
def _restore_ctype_maps():
    """Snapshot and restore CTYPE_MAPS so tests don't pollute the global registry."""
    snapshot = dict(CTYPE_MAPS)
    yield
    CTYPE_MAPS.clear()
    CTYPE_MAPS.update(snapshot)


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


def test_register_records_explicit_alignment():
    """Registering with alignof records the C++ alignment requirement."""
    cxx_name = "SomeLibrary::Aligned_abc123"
    numba_type = nbtypes.Opaque("SomeLibrary::Aligned_abc123")

    register_cxx_type(cxx_name, numba_type, alignof=32)

    result = to_numba_type(cxx_name)
    assert result is numba_type
    assert result.alignof_ == 32


@pytest.mark.parametrize("alignof", [0, 3, -8])
def test_register_rejects_invalid_alignment(alignof):
    """Explicit alignment must be a positive power of two."""
    cxx_name = f"SomeLibrary::InvalidAlign_{alignof}_abc123"
    numba_type = nbtypes.Opaque(cxx_name)

    with pytest.raises(ValueError, match="positive power of two"):
        register_cxx_type(cxx_name, numba_type, alignof=alignof)


def test_register_rejects_conflicting_alignment_metadata():
    """Existing type alignment metadata must not be silently overwritten."""
    numba_type = nbtypes.Opaque("SomeLibrary::ConflictingAligned_abc123")
    register_cxx_type("SomeLibrary::Aligned16_abc123", numba_type, alignof=16)

    with pytest.raises(ValueError, match="already has alignof_=16"):
        register_cxx_type(
            "SomeLibrary::Aligned32_abc123",
            numba_type,
            alignof=32,
        )
