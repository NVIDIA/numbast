# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the dependent/incomplete type guards in
``ast_canopy/cpp/src/record.cpp``.

Three related changes in this file:

1. ``Record::Record`` calls ``ctx.getTypeSize`` / ``ctx.getTypeAlign``
   on class template specialisations reaching the matcher with
   ``ANCESTOR_IS_NOT_TEMPLATE``. If the specialisation is still
   dependent or incomplete, Clang aborts inside the size query. The
   fix guards with ``!type->isDependentType() &&
   !type->isIncompleteType()`` and emits ``INVALID_SIZE_OF`` /
   ``INVALID_ALIGN_OF`` sentinels when layout cannot be computed.

2. Each per-child construction (``Field``, ``Method``,
   ``FunctionTemplate``, nested ``ClassTemplate``, nested ``Record``)
   is wrapped in ``try { ... } catch (...) { /* skip */ }`` so a
   single bad child does not lose the entire parent record.

3. Adds a ``<limits>`` include. ``INVALID_SIZE_OF`` /
   ``INVALID_ALIGN_OF`` use ``std::numeric_limits``; the header was
   transitively included in conda builds but missing in the manylinux
   wheel container. This cannot be exercised from pytest (it is a
   build-time concern); this file documents it for provenance.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(scope="module")
def source_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", "sample_record_incomplete.cu")


def test_incomplete_specialization_does_not_abort(source_path):
    """Header containing an incomplete class template specialisation
    (Fwd<int>) must not abort the parse."""
    decls = parse_declarations_from_source(
        source_path, [source_path], "sm_80", bypass_parse_error=True,
    )
    assert decls is not None


def test_wrapper_structs_parsed(source_path):
    """UsesIncomplete and UsesComplete should both parse."""
    decls = parse_declarations_from_source(
        source_path, [source_path], "sm_80", bypass_parse_error=True,
    )
    names = [s.name for s in decls.structs]
    assert "UsesIncomplete" in names, names
    assert "UsesComplete" in names, names


def test_complete_specialization_has_layout(source_path):
    """The fix must not regress layout computation for complete
    specialisations: Complete<float> still gets a valid sizeof_."""
    decls = parse_declarations_from_source(
        source_path, [source_path], "sm_80", bypass_parse_error=True,
    )
    complete_specs = [
        cts
        for cts in decls.class_template_specializations
        if "Complete" in cts.qual_name
    ]
    assert complete_specs, "Complete<float> not in parsed specializations"
    assert complete_specs[0].sizeof_ > 0, complete_specs[0].sizeof_
