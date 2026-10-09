# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit TableGen op definitions from backend-neutral function plans."""

import re

from numbast.backends.tablegen.config import OutParamPolicy, TableGenOptions
from numbast.backends.tablegen.types import TypeMapper, _is_pointer, _pointee

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


def _looks_like_out_param(spelling: str, mapper: TypeMapper) -> bool:
    """C out-pointer heuristic. Conservative: an **allowlist**, not a denylist.

    ``spelling`` must be the qualifier-preserving spelling (``Type.name``), not
    ``unqualified_non_ref_type_name`` -- the latter strips ``const``, which
    would make every ``const Foo *`` input look like an out-parameter.

    Out-parameters are:

    * a non-const pointer to a **numeric scalar or enum** -- ``CUdevice *``,
      ``size_t *``, ``int *``;
    * a non-const pointer **to a handle** -- ``CUstream_st **``.

    Everything else stays an input: by-value types, ``void *`` / ``char *``
    buffers, ``void **`` argument arrays, and pointers to complete structs
    (``const CUDA_RESOURCE_DESC *`` is a descriptor you pass *in*).
    """
    if not _is_pointer(spelling) or "const" in spelling:
        return False

    pointee = _pointee(spelling)

    if pointee.rstrip().endswith("*"):
        # pointer-to-pointer: an out-handle only if the inner type is an opaque
        # record, not a data-pointer array like ``void **kernelParams``.
        inner = _pointee(pointee)
        if inner in ("void", "char") or inner in mapper.records:
            return False
        return mapper.handle_type(inner) != mapper.ptr_type()

    if pointee in ("void", "char"):
        return False
    if pointee in mapper.records:
        return False  # pointer to a complete struct: an in-descriptor

    mapped = mapper.map(pointee)
    # Only scalars and enums qualify; anything landing on the catch-all
    # pointer type is an input buffer.
    return bool(mapped) and mapped != mapper.ptr_type()


def classify_params(decl, mapper: TypeMapper, policy: str):
    """Tag each parameter ``"in"`` or ``"out"``, in declaration order.

    Shared by the op renderer, which *splits* on the tag, and the lowering
    table, which *keeps* the order and carries the tag. They must agree: the
    lowering is what allocas a slot for each out-parameter and loads it back,
    so a parameter the op exposes as a result and the lowering treats as an
    input (or the reverse) is a miscompile.
    """
    out = []
    for param, ptype in zip(decl.params, decl.param_types):
        is_out = (
            policy == OutParamPolicy.C_OUT_PARAMS
            and _looks_like_out_param(
                # Type.name keeps qualifiers; unqualified_non_ref_type_name drops
                # const, which the heuristic needs.
                ptype.name,
                mapper,
            )
        )
        out.append((param, ptype, "out" if is_out else "in"))
    return out


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

    def _split_params(self, decl):
        """Partition parameters into (inputs, out-params) by policy."""
        tagged = classify_params(decl, self._mapper, self._cfg.out_param_policy)
        inputs = [(p, t) for p, t, d in tagged if d == "in"]
        outs = [(p, t) for p, t, d in tagged if d == "out"]
        return inputs, outs

    def _render_one(self, plan) -> str:
        decl = plan.decl
        cfg = self._cfg
        inputs, outs = self._split_params(decl)

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
