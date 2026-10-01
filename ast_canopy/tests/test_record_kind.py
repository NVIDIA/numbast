# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle

from ast_canopy import api, parse_declarations_from_source


def _native_records(data_folder, monkeypatch):
    """The pybind ``Record`` objects, before they are wrapped in ``Struct``.

    Captured from the real parse rather than built from a hand-rolled command
    line, which would duplicate the driver-flag logic in :mod:`ast_canopy.api`.
    """
    captured = []
    original = api.bindings.parse_declarations_from_command_line

    def spy(*args, **kwargs):
        decls = original(*args, **kwargs)
        captured.append(decls)
        return decls

    monkeypatch.setattr(
        api.bindings, "parse_declarations_from_command_line", spy
    )
    srcstr = str(data_folder / "record_kind.cu")
    parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    assert captured, "parse did not go through the native entry point"
    return {r.name: r for r in captured[0].records}


def _records(data_folder):
    srcstr = str(data_folder / "record_kind.cu")
    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")
    return {s.name: s for s in decls.structs}


def test_union_is_distinguished_from_struct(data_folder):
    """A union and a struct are not interchangeable downstream.

    A consumer mapping records onto another type system needs to know which
    it has: a union's fields overlap, so it has no direct equivalent in a
    system that only has product types.
    """
    records = _records(data_folder)
    assert records["PlainUnion"].is_union is True
    assert records["PlainStruct"].is_union is False
    assert records["PlainClass"].is_union is False


def test_nested_record_kinds_are_distinguished(data_folder):
    """Nesting preserves the kind rather than collapsing it to one answer.

    All three kinds are checked, not just the union: a test that looked at a
    nested union alone would also pass against an implementation that reported
    every nested record as a union, or that inherited the flag from the parent.
    Comparing the whole mapping rather than individual keys additionally
    catches a nested record going missing.
    """
    parent = _records(data_folder)["WithNested"]

    # ``nested_records`` also reports the parent itself -- C++ gives every
    # class an injected-class-name, which is a member type. Note that it is
    # only reported for ``struct`` and ``union``: in a ``class`` the injected
    # name is private, and non-public members are filtered out. That asymmetry
    # is incidental to the kind flag, so it is dropped here rather than pinned.
    nested = {
        r.name: r.is_union
        for r in parent.nested_records
        if r.name != parent.name
    }

    assert nested == {
        "NestedUnion": True,
        "NestedStruct": False,
        "NestedClass": False,
    }


def test_is_union_survives_a_pickle_round_trip(data_folder, monkeypatch):
    """The native ``Record`` is picklable, and the kind has to survive it.

    Losing the flag here would silently turn every union back into a struct
    on the far side, which is a wrong answer rather than an error.

    Note this pickles the *pybind* ``Record``, not the Python ``Struct``
    wrapper -- the wrapper pickles its ``__dict__`` and would preserve the
    flag no matter what the native pickle did.
    """
    records = _native_records(data_folder, monkeypatch)

    # Both kinds must be present or the assertion below proves nothing.
    assert records["PlainUnion"].is_union
    assert not records["PlainStruct"].is_union

    for name, record in records.items():
        assert pickle.loads(pickle.dumps(record)).is_union == record.is_union, (
            name
        )


def test_nested_record_kinds_survive_a_pickle_round_trip(
    data_folder, monkeypatch
):
    """Nested records reach the far side through their own code path.

    ``__setstate__`` rebuilds them by casting the nested vector, so a nested
    record's kind is restored by a separate recursive step rather than by the
    field read that restores the parent's. Checking only top-level records
    would leave that step unexercised.
    """
    parent = _native_records(data_folder, monkeypatch)["WithNested"]
    before = {r.name: r.is_union for r in parent.nested_records}

    restored = pickle.loads(pickle.dumps(parent))
    after = {r.name: r.is_union for r in restored.nested_records}

    # A union among them, or the comparison would hold for the wrong reason.
    assert any(before.values()), before
    assert after == before
