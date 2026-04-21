# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the null-QualType guard in
``ast_canopy/cpp/src/type.cpp``.

Under ``bypass_parse_error=True`` parsing of headers that use deeply
dependent types (e.g. ``typename T::nested`` fields inside uninstantiated
templates), Clang can hand the ast_canopy ``Type`` constructor a null
``QualType``. The subsequent calls to ``qualtype.getAsString()`` and
``qualtype.getCanonicalType()`` then segfault.

The fix adds an early-return guard that substitutes a ``"<null-type>"``
placeholder, so parsing continues instead of crashing.

This test is defensive: the null-QualType code path is triggered by
Clang's internal error recovery and is not easy to force
deterministically from plain C++. The test therefore verifies that
parsing a header with dependent member types completes without crashing
the Python process and that non-dependent declarations remain
parseable.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_null_qualtype.cu")


def test_dependent_member_types_do_not_crash(source_path):
    """Parsing must not segfault on deeply dependent member types even
    when Clang's type resolution falls back to null QualTypes."""
    decls = parse_declarations_from_source(
        source_path, [source_path], "sm_80", bypass_parse_error=True,
    )
    assert decls is not None


def test_plain_struct_still_parsed(source_path):
    """The presence of dependent/null-type fields must not prevent
    surrounding non-dependent structs from being reported."""
    decls = parse_declarations_from_source(
        source_path, [source_path], "sm_80", bypass_parse_error=True,
    )
    names = [s.name for s in decls.structs]
    assert "Plain" in names, names
