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
``Typedef.underlying_name`` the canonical one. All this module does is index
them the way the renderers ask the question -- given a tag, what does the
library call it -- and note which public names denote enums, because an enum
lowers to an integer and anything unrecognised lowers to an opaque pointer.
"""

import re
from dataclasses import dataclass, field

__all__ = ["TypedefFacts", "typedef_facts"]

#: ``struct AcmeStream_st *`` -> ``AcmeStream_st``. The elaborated keyword and
#: any pointer depth are spelling, not identity.
_TAG = re.compile(r"^(?:struct|union|enum|class)\s+(\w+)\s*\**$")


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
    """Index ``decls.typedefs`` by the question the renderers ask."""
    facts = TypedefFacts()

    for typedef in decls.typedefs:
        underlying = typedef.underlying_name.strip()

        match = _TAG.match(underlying)
        if match:
            # setdefault: several typedefs may name one tag, and the first is
            # the one the header leads with.
            facts.tag_to_public.setdefault(match.group(1), typedef.name)

        if underlying.startswith("enum "):
            facts.enum_public_names.add(typedef.name)

    return facts
