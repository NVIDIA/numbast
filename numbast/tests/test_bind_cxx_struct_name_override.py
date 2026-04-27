# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone test for the ``name=`` override on ``bind_cxx_struct``.

For a class template specialization, ``struct_decl.name`` is just the
unqualified template name (e.g. ``"Vec"``), but the Numba type name,
CTYPE_MAPS key, and shim code all need the fully qualified
specialization (e.g. ``"demo::Vec<float, 3>"``). The ``name=`` keyword
on ``bind_cxx_struct`` lets callers supply the fully-qualified name.
"""

import os

import pytest
from numba import types as nbtypes
from numba.cuda.datamodel.models import StructModel

from ast_canopy import parse_declarations_from_source
from numbast import bind_cxx_struct, MemoryShimWriter
from numbast.types import CTYPE_MAPS, to_numba_type


@pytest.fixture(autouse=True)
def _restore_ctype_maps():
    """Snapshot and restore CTYPE_MAPS so bind_cxx_struct side effects
    don't leak across tests."""
    snapshot = dict(CTYPE_MAPS)
    yield
    CTYPE_MAPS.clear()
    CTYPE_MAPS.update(snapshot)


def _parse_specialization():
    header = os.path.join(
        os.path.dirname(__file__), "data", "sample_name_override.cuh"
    )
    decls = parse_declarations_from_source(header, [header], "sm_80")
    specs = [
        cts for cts in decls.class_template_specializations if "Vec" in cts.name
    ]
    assert specs, "expected Vec<float, 3> specialization to be parsed"
    return specs[0]


def test_specialization_name_is_unqualified_by_default():
    """Guard: struct_decl.name on the parsed specialization is the short
    template name. This is the reason the override is needed."""
    cts = _parse_specialization()
    assert cts.name == "Vec"


def test_bind_without_name_uses_struct_decl_name():
    """Without ``name=``, numba type name and CTYPE_MAPS key use the
    unqualified specialization name from struct_decl."""
    cts = _parse_specialization()
    shim_writer = MemoryShimWriter("")
    S = bind_cxx_struct(shim_writer, cts, nbtypes.Type, StructModel)

    assert S._nbtype.name == "Vec"
    assert CTYPE_MAPS.get("Vec") is S._nbtype
    assert to_numba_type("Vec") is S._nbtype


def test_bind_with_name_uses_override_everywhere():
    """With ``name=``, the override drives Numba type name, Python API
    class name, and the CTYPE_MAPS registration key."""
    cts = _parse_specialization()
    shim_writer = MemoryShimWriter("")
    override = "demo::Vec<float, 3>"

    S = bind_cxx_struct(
        shim_writer,
        cts,
        nbtypes.Type,
        StructModel,
        name=override,
    )

    assert S._nbtype.name == override
    assert S.__name__ == override
    assert CTYPE_MAPS.get(override) is S._nbtype
    assert to_numba_type(override) is S._nbtype


def test_bind_with_name_honors_aliases_from_typedef_names():
    """Typedef-derived aliases are keyed by the parsed declaration name.

    ``name=`` changes the effective C++ type name used by the binding, but it
    must not drop aliases produced from typedefs of the original parsed name.
    """
    cts = _parse_specialization()
    shim_writer = MemoryShimWriter("")
    override = "demo::Vec<float, 3>"
    aliases = {
        "Vec": ["VecAliasFromTypedef"],
        override: ["VecAliasFromOverride"],
    }

    S = bind_cxx_struct(
        shim_writer,
        cts,
        nbtypes.Type,
        StructModel,
        aliases=aliases,
        name=override,
    )

    assert to_numba_type("VecAliasFromTypedef") is S._nbtype
    assert to_numba_type("VecAliasFromOverride") is S._nbtype


def test_sanitize_c_identifier_strips_template_syntax():
    """When a fully-qualified template specialization is passed via
    ``name=``, it is embedded in the conversion-operator shim symbol.
    ``_sanitize_c_identifier`` must rewrite every character outside
    ``[A-Za-z0-9_]`` (``:``, ``<``, ``>``, ``,``, spaces) to keep the
    resulting C symbol valid."""
    import re

    from numbast.struct import _sanitize_c_identifier

    result = _sanitize_c_identifier("Eigen::Matrix<float, 3, 1>")
    assert re.fullmatch(r"[A-Za-z0-9_]+", result)
    assert "<" not in result and ">" not in result
    assert ":" not in result and "," not in result
