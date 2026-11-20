# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit ``<Prefix>Ops.td``: one op definition per curated entry point.

Formatting only. Which ops exist, what their operands and results are, and
what everything is called were all settled in
:mod:`numbast.backends.tablegen.plan`.
"""

from numbast.backends.tablegen.model import DialectSpec, OpSpec

__all__ = ["render_ops_td"]

_OPS_HEADER = """\
#ifndef {guard}_OPS_TD
#define {guard}_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "{dialect}_dialect/Dialect/{prefix}/IR/{prefix}Dialect.td"
include "{dialect}_dialect/Dialect/{prefix}/IR/{prefix}Types.td"

class {prefix}_Op<string mnemonic, list<Trait> traits = []>
    : Op<{prefix}_Dialect, mnemonic, traits>;
"""

_OPS_FOOTER = """
#endif // {guard}_OPS_TD
"""

_OP_TEMPLATE = """\
def {prefix}_{record}Op
    : {prefix}_Op<"{mnemonic}", []> {{

  let arguments = (ins{arguments});
  let results = (outs{results});
  let extraClassDeclaration = [{{
    std::string getAPI() {{ return std::string("{shim}"); }}
  }}];
}}
"""


def _operand_list(operands) -> str:
    if not operands:
        return ""
    return " " + ", ".join(f"{o.type_ref}:${o.name}" for o in operands)


def _render_one(op: OpSpec, dialect: DialectSpec) -> str:
    return _OP_TEMPLATE.format(
        prefix=dialect.cpp_class_prefix,
        record=op.record,
        mnemonic=op.mnemonic,
        arguments=_operand_list(op.operands),
        results=_operand_list(op.results),
        shim=op.shim,
    )


def render_ops_td(ops, dialect: DialectSpec) -> str:
    """All op definitions, in declaration order.

    Wrapped in the preamble that makes the file tablegen-able on its own:
    the include guard, the ``.td`` files the definitions below refer to,
    and the ``{prefix}_Op`` base class every definition derives from.
    Without it mlir-tblgen cannot resolve the base class, which no amount
    of comparing op signatures will reveal.
    """
    fields = {
        "guard": dialect.guard,
        "prefix": dialect.cpp_class_prefix,
        "dialect": dialect.name,
    }
    body = "\n".join(_render_one(op, dialect) for op in ops)
    return (
        _OPS_HEADER.format(**fields)
        + "\n"
        + body
        + _OPS_FOOTER.format(**fields)
    )
