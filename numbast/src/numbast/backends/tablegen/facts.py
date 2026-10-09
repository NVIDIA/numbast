# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the renderers need to know about a header beyond its functions."""

from numbast.backends.tablegen.typedefs import recover_typedefs

__all__ = ["header_facts"]


def header_facts(decls, header_path):
    """Everything the renderers need about the header beyond the functions.

    Public because the enum set in particular is easy to get wrong by hand:
    a signature may name an enum by its tag or by its typedef, ast_canopy
    reports neither reliably, and a name missing from this set silently maps
    to the opaque pointer type instead of an integer.
    """
    typedefs = recover_typedefs(header_path)
    enums = (
        {e.name for e in decls.enums}
        | set(typedefs.enum_tag_to_public)
        | typedefs.enum_names()
    )
    return {
        "records": {s.name for s in decls.structs},
        "record_objs": {s.name: s for s in decls.structs},
        "unions": {s.name for s in decls.structs if s.is_union},
        "enums": enums,
        "typedefs": typedefs,
    }
