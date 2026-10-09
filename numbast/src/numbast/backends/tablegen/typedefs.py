# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recover the public type vocabulary that ``ast_canopy`` does not report.

WORKAROUND -- see :mod:`numbast.backends.tablegen` docs.

``ast_canopy`` hands back *canonical* spellings, so a header written as::

    typedef struct AcmeStream_st *AcmeStream;
    typedef enum AcmeResult_enum { ... } AcmeResult;

    AcmeResult acmeStreamCreate(AcmeStream *phStream, unsigned int Flags);

is reported as ``ret='AcmeResult_enum'`` and
``params=[('phStream', 'AcmeStream_st * *'), ...]``, and
``parse_declarations_from_source(...).typedefs`` comes back **empty** for both
of those typedef forms. The library's public names -- the ones a dialect's types
must be named after -- are gone.

This matters much less for the Numba backends, which map a spelling to a Numba
type through a flat table. It matters a great deal for a dialect, where the type
*names* are the public API: without recovery, ``AcheStream`` becomes
``Acme_Stream_stType`` rather than ``Acme_StreamType``.

Until ``ast_canopy`` exposes pointer and enum typedefs, recover them from the
header text. The cuda-python dialect generator carries the same workaround
(``autogen/ingest.py::_build_typedef_maps``, "Recover the typedef facts
ast_canopy drops"), which is the strongest argument for fixing it upstream
instead.
"""

import re
from dataclasses import dataclass, field

__all__ = ["TypedefFacts", "recover_typedefs"]


@dataclass
class TypedefFacts:
    """Public names recovered for record tags and enum tags."""

    tag_to_public: dict[str, str] = field(default_factory=dict)
    """``AcmeStream_st`` -> ``AcmeStream`` (pointer-to-incomplete-struct handles)."""

    enum_tag_to_public: dict[str, str] = field(default_factory=dict)
    """``AcmeResult_enum`` -> ``AcmeResult``."""

    anonymous_enums: set = field(default_factory=set)
    """Public names of ``typedef enum { ... } Public;`` -- no tag to map from.

    ast_canopy does not report these at all, so without recovering them the
    backend never learns they are enums and maps them to the opaque pointer
    type instead of an integer."""

    def enum_names(self) -> set:
        """Every public enum name this recovered, tagged or not."""
        return set(self.enum_tag_to_public.values()) | self.anonymous_enums

    def public(self, name: str) -> str:
        """Public spelling for ``name``, or ``name`` itself if not recovered."""
        return self.tag_to_public.get(
            name, self.enum_tag_to_public.get(name, name)
        )


def recover_typedefs(header_path: str) -> TypedefFacts:
    """Parse ``header_path`` textually for typedef forms ast_canopy drops."""
    with open(header_path) as f:
        text = f.read()
    return recover_typedefs_from_text(text)


def recover_typedefs_from_text(text: str) -> TypedefFacts:
    facts = TypedefFacts()

    # typedef struct Tag *Public;
    for tag, public in re.findall(
        r"typedef\s+struct\s+(\w+)\s*\*\s*(\w+)\s*;", text
    ):
        facts.tag_to_public.setdefault(tag, public)

    # typedef enum Tag { ... } Public;
    for tag, public in re.findall(
        r"typedef\s+enum\s+(\w+)\s*\{[^}]*\}\s*(\w+)\s*;", text, re.S
    ):
        facts.enum_tag_to_public.setdefault(tag, public)

    # typedef enum { ... } Public;  -- no tag, so nothing to map, but the name
    # still has to be known to be an enum or it maps to a pointer.
    for public in re.findall(
        r"typedef\s+enum\s*\{[^}]*\}\s*(\w+)\s*;", text, re.S
    ):
        facts.anonymous_enums.add(public)

    return facts
