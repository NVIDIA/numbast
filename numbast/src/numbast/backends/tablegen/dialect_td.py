# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit ``<Prefix>Dialect.td``: the dialect definition itself.

The smallest of the three artifacts and the one every other generated file
includes. It declares the mnemonic namespace an op's name is resolved in
(``acme`` -> ``acme.stream.create``), the C++ namespace the generated classes
land in, and that the dialect parses and prints its own types.
"""

from numbast.backends.tablegen.model import DialectSpec

__all__ = ["render_dialect_td"]

_DIALECT_TD = """\
#ifndef {guard}_DIALECT_TD
#define {guard}_DIALECT_TD

include "mlir/IR/DialectBase.td"

def {prefix}_Dialect : Dialect {{
  let name = "{dialect}";
  let summary = "{summary}";
  let description = [{{
    {description}
  }}];
  let cppNamespace = "{cpp_namespace}";
  let useDefaultTypePrinterParser = 1;
}}

#endif // {guard}_DIALECT_TD
"""


def render_dialect_td(dialect: DialectSpec) -> str:
    """The dialect definition every other generated file includes."""
    return _DIALECT_TD.format(
        guard=dialect.guard,
        prefix=dialect.cpp_class_prefix,
        dialect=dialect.name,
        summary=dialect.summary,
        description=dialect.description,
        cpp_namespace=dialect.cpp_namespace,
    )
