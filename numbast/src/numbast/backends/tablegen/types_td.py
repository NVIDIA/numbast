# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit ``<Prefix>Types.td``: the dialect's type vocabulary.

Formatting only. Which types the dialect declares is decided in
:mod:`numbast.backends.tablegen.plan`, where resolving the ops discovers them.
"""

from numbast.backends.tablegen.model import DialectSpec

__all__ = ["render_types_td"]

_TYPES_HEADER = """\
#ifndef {guard}_TYPES_TD
#define {guard}_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "{dialect}_dialect/Dialect/{prefix}/IR/{prefix}Dialect.td"

class {prefix}_Type<string name, string typeMnemonic>
    : TypeDef<{prefix}_Dialect, name> {{
  let mnemonic = typeMnemonic;
}}
"""

_TYPE_DEF = """
def {prefix}_{record}Type : {prefix}_Type<"{record}", "{mnemonic}"> {{
  let summary = "{summary}";
  let description = [{{
    {summary}
  }}];
}}
"""

_TYPES_FOOTER = """
#endif // {guard}_TYPES_TD
"""


def render_types_td(types, dialect: DialectSpec) -> str:
    """One ``TypeDef`` per entry, in the order analysis put them in."""
    out = [
        _TYPES_HEADER.format(
            guard=dialect.guard,
            prefix=dialect.cpp_class_prefix,
            dialect=dialect.name,
        )
    ]

    out += [
        _TYPE_DEF.format(
            prefix=dialect.cpp_class_prefix,
            record=t.record,
            mnemonic=t.mnemonic,
            summary=t.summary,
        )
        for t in types
    ]

    out.append(_TYPES_FOOTER.format(guard=dialect.guard))
    return "".join(out)
