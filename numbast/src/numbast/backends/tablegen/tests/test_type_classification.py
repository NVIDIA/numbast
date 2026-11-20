# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the mapper decides a spelling is, before anything is named.

``expected/`` pins the names. These pin the decision behind them, which is a
different thing: a spelling can be classified wrongly and still come out with
a plausible-looking name, and then the only evidence is a type in the
vocabulary with no C declaration behind it.
"""

import pytest

from numbast.backends.tablegen.type_mapper import TypeKind, TypeMapper

#: Enough header facts to exercise every branch: one complete record, one
#: enum, and -- by their absence -- one incomplete tag.
MAPPER = TypeMapper(
    "Acme",
    handle_prefix="Acme",
    struct_prefixes=["Acme"],
    records={"AcmeDesc_st"},
    enums={"AcmeResult_enum"},
)


@pytest.mark.parametrize(
    "spelling,kind,subject",
    [
        # Scalars and enums resolve to builtins the dialect need not declare.
        ("unsigned int", TypeKind.BUILTIN, "unsigned int"),
        ("float", TypeKind.BUILTIN, "float"),
        ("AcmeResult_enum", TypeKind.BUILTIN, "AcmeResult_enum"),
        ("void", TypeKind.VOID, "void"),
        # A tag with a body is a record; the same tag behind a pointer is the
        # same record, because the dialect models it by value either way.
        ("AcmeDesc_st", TypeKind.RECORD, "AcmeDesc_st"),
        ("AcmeDesc_st *", TypeKind.RECORD, "AcmeDesc_st"),
        # A tag with no body, behind a pointer, is an opaque handle. This is
        # the only signal the header gives that it is one.
        ("AcmeStream_st *", TypeKind.HANDLE, "AcmeStream_st"),
        # The floor: nothing here can be given a meaningful name.
        ("void *", TypeKind.OPAQUE, "void"),
        ("char *", TypeKind.OPAQUE, "char"),
        ("AcmeStream_st * *", TypeKind.OPAQUE, "AcmeStream_st *"),
        ("void (*)(void *)", TypeKind.OPAQUE, "void (*)(void *)"),
        # Floats are known by spelling rather than from the integer-width
        # table, so a pointer to one has to be recognised against both. Against
        # only the table it reaches the handle rule and mints !acme.float, a
        # type whose actual meaning is "pointer to float".
        ("float *", TypeKind.OPAQUE, "float"),
        ("double *", TypeKind.OPAQUE, "double"),
    ],
)
def test_spelling_classifies_as(spelling, kind, subject):
    what = MAPPER.classify(spelling)
    assert what.kind is kind
    assert what.subject == subject


def test_classifying_mints_nothing():
    """Deciding is separable from naming, and stays that way.

    The registries are the mapper's second output, so a classification that
    quietly added to them would make the decision untestable in isolation --
    and would put types in the dialect that no signature asked for.
    """
    mapper = TypeMapper("Acme", handle_prefix="Acme", records={"AcmeDesc_st"})

    for spelling in ("AcmeStream_st *", "AcmeDesc_st *", "float *"):
        mapper.classify(spelling)

    assert mapper.handles == {}
    assert mapper.structs == {}
