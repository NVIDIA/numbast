# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public type vocabulary a C library declares through its typedefs.

A C library uses typedefs to separate its public vocabulary from its
implementation vocabulary: ``CUstream`` is the API, and ``CUstream_st`` is an
incomplete struct that exists only so the pointer has something to point at.
A front-end is right to canonicalise the two -- they are the same type -- but a
dialect's type *names* are its interface, so a generator that only sees
canonical spellings would mint ``!cudadrv.streamst``.

``ast_canopy`` reports both halves: ``Typedef.name`` is the public name and
``Typedef.underlying_type`` the canonical one. All this module does is index
them the way the renderers ask the question -- given a tag, what does the
library call it -- and note which public names denote enums, because an enum
lowers to an integer and anything unrecognised lowers to an opaque pointer.
"""

from dataclasses import dataclass, field

from ast_canopy.pylibastcanopy import type_kind

__all__ = ["TypedefFacts", "typedef_facts"]

#: Underlying kinds that carry a tag worth indexing. A pointer is included
#: because an opaque handle is a typedef of a *pointer* to an incomplete
#: record, which is the form the tag has to be recovered through.
_TAGGED = frozenset({type_kind.record, type_kind.enum_, type_kind.pointer})


@dataclass
class TypedefFacts:
    """Public names recovered for record and enum tags."""

    tag_to_public: dict[str, str] = field(default_factory=dict)
    """``AcmeStream_st`` -> ``AcmeStream``, for every tag a typedef names."""

    enum_public_names: set[str] = field(default_factory=set)
    """Public names whose underlying type is an enum.

    Kept separately because enum-ness decides the lowered type: an enum is an
    integer, and a name the mapper does not recognise falls through to the
    dialect's opaque pointer type.
    """

    def public(self, name: str) -> str:
        """Public spelling for ``name``, or ``name`` itself if it has none."""
        return self.tag_to_public.get(name, name)


def typedef_facts(decls) -> TypedefFacts:
    """Index ``decls.typedefs`` by the question the renderers ask.

    Keyed off ``Type.kind`` rather than the printed spelling: whether the
    elaborated keyword appears in a type's name is a choice clang makes about
    formatting, while the kind is what the type *is*.
    """
    facts = TypedefFacts()

    for typedef in decls.typedefs:
        underlying = typedef.underlying_type

        if underlying.kind in _TAGGED:
            # The name still carries pointer depth -- an opaque handle's
            # underlying type prints as ``AcmeStream_st *``.
            tag = underlying.name.rstrip("* ").strip()
            if tag:
                # setdefault: several typedefs may name one tag, and the first
                # is the one the header leads with.
                facts.tag_to_public.setdefault(tag, typedef.name)

        if underlying.kind == type_kind.enum_:
            facts.enum_public_names.add(typedef.name)

    return facts
