# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit ``<Prefix>Ops.td``: one op definition per curated entry point.

Consumes exactly what :mod:`numbast.curate` produces -- the declaration
plus the exposed name -- and adds only emission concerns: type mapping,
out-parameter lifting, and the status result.
"""

import re

from numbast.backends.tablegen.config import OutParamPolicy, TableGenOptions
from numbast.backends.tablegen.type_mapper import TypeMapper
from numbast.intent_defs import ArgIntent

__all__ = [
    "TableGenFunctionsRenderer",
    "classify_params",
    "mnemonic_for",
    "sanitize_operand_name",
]

# Names that collide with the accessors mlir-tblgen generates on Operation/Op.
# Emitting an operand or result with one of these names makes mlir-tblgen fail,
# so they must be renamed. No equivalent constraint exists for Numba emission.
_RESERVED = frozenset(
    {
        "attributes",
        "block",
        "context",
        "dialect",
        "loc",
        "location",
        "name",
        "operands",
        "operation",
        "properties",
        "regions",
        "result",
        "results",
        "results_",
        "successors",
        "type",
        "types",
        "value",
        "values",
    }
)


def sanitize_operand_name(name: str, taken=()) -> str:
    """Make ``name`` legal as a TableGen operand/result name and unique."""
    out = name or "arg"
    while out in _RESERVED or out in taken:
        out += "_"
    return out


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


# Camel-case splitter. Digit runs are their own component, so `WaitValue32`
# splits as wait/value/32 rather than wait/value32 -- otherwise the same digits
# split inconsistently depending on the preceding letter (`MemsetD2D16` has no
# lowercase to absorb them, `WaitValue32` does), and the dialect would carry
# both `memset.d.2.d.16` and `wait.value32`.
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+")


def mnemonic_for(exposed_name: str) -> str:
    """``streamWaitEvent`` -> ``stream.wait.event``.

    Splits on camel-case and underscores and joins with dots, which is the
    conventional MLIR spelling for a namespaced operation.
    """
    return ".".join(p.lower() for p in _WORD.findall(exposed_name) if p)


def _record_for(exposed_name: str) -> str:
    """``streamWaitEvent`` -> ``StreamWaitEvent`` (the TableGen record stem)."""
    parts = _WORD.findall(exposed_name)
    return "".join(
        p[:1].upper() + p[1:].lower() if p.isupper() else p[:1].upper() + p[1:]
        for p in parts
    )


def classify_params(decl, intents, policy: str):
    """Tag each parameter ``"in"`` or ``"out"``, in declaration order.

    Shared by the op renderer, which *splits* on the tag, and the lowering
    table, which *keeps* the order and carries the tag. They must agree: the
    lowering is what allocas a slot for each out-parameter and loads it back,
    so a parameter the op exposes as a result and the lowering treats as an
    input (or the reverse) is a miscompile. That is why this is one function
    rather than a rule each renderer applies for itself.

    ``intents`` is this function's entry from ``Function Argument Intents``,
    keyed by parameter name or 0-based index, as the Numba backends key it.
    """
    return [
        (param, ptype, _direction(param, index, intents, policy))
        for index, (param, ptype) in enumerate(
            zip(decl.params, decl.param_types)
        )
    ]


def _direction(param, index: int, intents, policy: str) -> str:
    """``"out"`` if this parameter was declared an out-return, else ``"in"``."""
    if policy != OutParamPolicy.EXPLICIT or not intents:
        return "in"

    # Name wins over index, matching compute_intent_plan's precedence.
    declared = intents.get(getattr(param, "name", None), intents.get(index))
    if declared is None:
        return "in"

    # ArgIntent is a str-valued enum, so a configured string converts directly,
    # and an unrecognised spelling raises rather than defaulting to "in" -- a
    # typo in a parameter's direction is otherwise a silent miscompile.
    return "out" if ArgIntent(declared) is ArgIntent.out_return else "in"


class TableGenFunctionsRenderer:
    """Render ``FunctionPlan`` objects as TableGen op definitions.

    Consumes exactly what :mod:`numbast.curate` produces -- the declaration
    plus the exposed name -- and adds only emission concerns: type mapping,
    out-parameter lifting, and the status result.
    """

    def __init__(
        self,
        plans,
        config: TableGenOptions,
        *,
        records: set[str] | None = None,
        enums: set[str] | None = None,
        typedefs=None,
        record_objs: dict | None = None,
    ):
        self._plans = list(plans)
        self._cfg = config
        self._typedefs = typedefs
        self._mapper = TypeMapper(
            config.cpp_class_prefix,
            handle_prefix=config.handle_prefix,
            struct_prefixes=config.struct_prefixes,
            records=records,
            enums=enums,
            typedefs=typedefs,
            record_objs=record_objs,
            opaque_structs=config.opaque_structs,
        )

    @property
    def mapper(self) -> TypeMapper:
        """Type mapper, including any handle/struct types minted while rendering."""
        return self._mapper

    def _split_params(self, plan):
        """Partition parameters into (inputs, out-params) by policy."""
        tagged = classify_params(
            plan.decl,
            plan.argument_intents.get(plan.c_symbol),
            self._cfg.out_param_policy,
        )
        inputs = [(p, t) for p, t, d in tagged if d == "in"]
        outs = [(p, t) for p, t, d in tagged if d == "out"]
        return inputs, outs

    def _render_one(self, plan) -> str:
        decl = plan.decl
        cfg = self._cfg
        inputs, outs = self._split_params(plan)

        taken: set[str] = set()

        args = []
        for param, ptype in inputs:
            mapped = self._mapper.map(ptype.unqualified_non_ref_type_name)
            if mapped:
                nm = sanitize_operand_name(param.name, taken)
                taken.add(nm)
                args.append(f"{mapped}:${nm}")

        results = []
        for param, ptype in outs:
            mapped = self._mapper.map_pointee(
                ptype.unqualified_non_ref_type_name
            )
            if mapped:
                nm = sanitize_operand_name(param.name, taken)
                taken.add(nm)
                results.append(f"{mapped}:${nm}")

        # A real (non-void, non-status) return value becomes a leading result.
        ret_spelling = decl.return_type.unqualified_non_ref_type_name.strip()
        ret = self._mapper.map(ret_spelling)
        # ast_canopy reports the enum *tag* (AcmeResult_enum), not the public
        # typedef the config names (AcmeResult), so compare through recovery.
        ret_public = (
            self._typedefs.public(ret_spelling)
            if self._typedefs is not None
            else ret_spelling
        )
        is_status = (
            cfg.status_type is not None and ret_public == cfg.status_type
        )
        if ret and not is_status:
            # "result" is reserved, so a real return value needs a free name.
            nm = next(
                c
                for c in ("result", "ret", "ret_code", "value_", "out")
                if c not in _RESERVED and c not in taken
            )
            taken.add(nm)
            results.insert(0, f"{ret}:${nm}")

        # Status trails, matching the dialect convention -- but only for entry
        # points that actually return it. A library where some functions return
        # a value instead must not get a phantom status result.
        if is_status:
            results.append(f"{cfg.cpp_class_prefix}_ResultType:$result")

        mnemonic = cfg.mnemonic_overrides.get(
            plan.exposed_name, mnemonic_for(plan.exposed_name)
        )
        return _OP_TEMPLATE.format(
            prefix=cfg.cpp_class_prefix,
            record=_record_for(plan.exposed_name),
            mnemonic=mnemonic,
            arguments=(" " + ", ".join(args)) if args else "",
            results=(" " + ", ".join(results)) if results else "",
            shim=f"{cfg.shim_prefix}{plan.c_symbol}",
        )

    def render_as_str(self) -> str:
        """All op definitions, in declaration order.

        Wrapped in the preamble that makes the file tablegen-able on its own:
        the include guard, the ``.td`` files the definitions below refer to,
        and the ``{prefix}_Op`` base class every definition derives from.
        Without it mlir-tblgen cannot resolve the base class, which no amount
        of comparing op signatures will reveal.
        """
        cfg = self._cfg
        fields = {
            "guard": cfg.dialect_name.upper(),
            "prefix": cfg.cpp_class_prefix,
            "dialect": cfg.dialect_name,
        }
        body = "\n".join(self._render_one(p) for p in self._plans)
        return (
            _OPS_HEADER.format(**fields)
            + "\n"
            + body
            + _OPS_FOOTER.format(**fields)
        )
