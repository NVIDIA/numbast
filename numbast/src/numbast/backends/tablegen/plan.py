# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve curated entry points into the dialect they imply.

Everything that requires a decision happens here: which parameters are
results, what each type is called, which types exist at all, and whether the
answer is coherent enough to emit. The renderers downstream only format.

Resolving the ops is also what discovers the types. A curated signature
reaches a handle or a record, that record reaches more records through its
fields, and the dialect has to declare every one of them -- so the type
vocabulary cannot be known until the last op has been resolved. The ordering
that follows from that is an invariant of this module rather than a caller's
responsibility, which is the reason :func:`plan_dialect` returns both halves
of the model together.
"""

import re
from collections import Counter

from numbast.backends.tablegen.config import OutParamPolicy, TableGenOptions
from numbast.backends.tablegen.model import (
    DialectModel,
    DialectSpec,
    OpSpec,
    Operand,
    TypeSpec,
)
from numbast.backends.tablegen.type_mapper import TypeMapper
from numbast.intent_defs import ArgIntent

__all__ = [
    "DuplicateTypeMnemonicError",
    "classify_params",
    "collect_types",
    "mnemonic_for",
    "plan_dialect",
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


class DuplicateTypeMnemonicError(Exception):
    """Two ``TypeDef``s claim one mnemonic.

    TableGen keys types on the C++ name, so this passes tblgen and compiles;
    the dialect then aborts the host process at registration with ``Dialect
    Type with name ... is already registered``. Record names that differ only
    in case collapse here, because a struct's mnemonic is its name lowered.

    Raised rather than resolved by renaming: the two names stand for two
    different layouts, and silently merging or renumbering them would put a
    guess in the type vocabulary.
    """


def sanitize_operand_name(name: str, taken=()) -> str:
    """Make ``name`` legal as a TableGen operand/result name and unique."""
    out = name or "arg"
    while out in _RESERVED or out in taken:
        out += "_"
    return out


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

    Shared by op resolution, which *splits* on the tag, and the lowering
    table, which *keeps* the order and carries the tag. They must agree: the
    lowering is what allocas a slot for each out-parameter and loads it back,
    so a parameter the op exposes as a result and the lowering treats as an
    input (or the reverse) is a miscompile. That is why this is one function
    rather than a rule each consumer applies for itself.

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
    declared = intents.get(param.name, intents.get(index))
    if declared is None:
        return "in"

    # ArgIntent is a str-valued enum, so a configured string converts directly,
    # and an unrecognised spelling raises rather than defaulting to "in" -- a
    # typo in a parameter's direction is otherwise a silent miscompile.
    return "out" if ArgIntent(declared) is ArgIntent.out_return else "in"


def _dialect_spec(options: TableGenOptions) -> DialectSpec:
    return DialectSpec(
        name=options.dialect_name,
        cpp_class_prefix=options.cpp_class_prefix,
        summary=f"{options.dialect_name} dialect",
        description=(
            f"Autogenerated FFI mirror of the {options.dialect_name} API."
        ),
    )


def _resolve_op(plan, options: TableGenOptions, mapper, typedefs) -> OpSpec:
    """One curated entry point, with its types minted into ``mapper``."""
    decl = plan.decl
    tagged = classify_params(
        decl,
        plan.argument_intents.get(plan.c_symbol),
        options.out_param_policy,
    )

    taken: set[str] = set()

    operands = []
    for param, ptype, direction in tagged:
        if direction != "in":
            continue
        mapped = mapper.map(ptype.unqualified_non_ref_type_name)
        if mapped:
            name = sanitize_operand_name(param.name, taken)
            taken.add(name)
            operands.append(Operand(mapped, name))

    results = []
    for param, ptype, direction in tagged:
        if direction != "out":
            continue
        mapped = mapper.map_pointee(ptype.unqualified_non_ref_type_name)
        if mapped:
            name = sanitize_operand_name(param.name, taken)
            taken.add(name)
            results.append(Operand(mapped, name))

    # A real (non-void, non-status) return value becomes a leading result.
    ret_spelling = decl.return_type.unqualified_non_ref_type_name.strip()
    ret = mapper.map(ret_spelling)
    # ast_canopy reports the enum *tag* (AcmeResult_enum), not the public
    # typedef the config names (AcmeResult), so compare through recovery.
    ret_public = (
        typedefs.public(ret_spelling) if typedefs is not None else ret_spelling
    )
    is_status = (
        options.status_type is not None and ret_public == options.status_type
    )
    if ret and not is_status:
        # "result" is reserved, so a real return value needs a free name.
        name = next(
            c
            for c in ("result", "ret", "ret_code", "value_", "out")
            if c not in _RESERVED and c not in taken
        )
        taken.add(name)
        results.insert(0, Operand(ret, name))

    # Status trails, matching the dialect convention -- but only for entry
    # points that actually return it. A library where some functions return
    # a value instead must not get a phantom status result.
    if is_status:
        results.append(
            Operand(f"{options.cpp_class_prefix}_ResultType", "result")
        )

    return OpSpec(
        record=_record_for(plan.exposed_name),
        mnemonic=options.mnemonic_overrides.get(
            plan.exposed_name, mnemonic_for(plan.exposed_name)
        ),
        operands=tuple(operands),
        results=tuple(results),
        shim=f"{options.shim_prefix}{plan.c_symbol}",
    )


def collect_types(
    options: TableGenOptions, mapper, unions: set = frozenset()
) -> tuple[TypeSpec, ...]:
    """The types the resolved ops reached, in emission order.

    Three kinds, after the status type the recipe may name: a single opaque
    pointer, the FFI floor for everything that is not a scalar, a handle or a
    modelled record; one type per handle; one type per modelled record.
    Records are nominal -- fields live in the descriptor the lowering
    consumes, not in TableGen, matching the reference dialect.

    Validates before returning, so no caller can hold an incoherent
    vocabulary.
    """
    types = []

    if options.status_type is not None:
        types.append(
            TypeSpec(
                record="Result",
                mnemonic="result",
                summary=f"{options.status_type} status code",
                kind="status",
            )
        )

    types.append(
        TypeSpec(
            record="Ptr",
            mnemonic="ptr",
            summary="opaque caller-owned pointer/handle (FFI)",
            kind="ptr",
        )
    )

    for record, mnemonic in sorted(mapper.handles.items()):
        types.append(
            TypeSpec(
                record=record,
                mnemonic=mnemonic,
                summary=f"opaque {options.dialect_name} {mnemonic} handle",
                kind="handle",
            )
        )

    for c_name, record in sorted(mapper.structs.items(), key=lambda kv: kv[1]):
        kind = "union" if c_name in unions else "struct"
        types.append(
            TypeSpec(
                record=record,
                mnemonic=record.lower(),
                summary=f"{c_name} ({kind})",
                kind="record",
            )
        )

    _reject_duplicate_mnemonics(types)
    return tuple(types)


def _reject_duplicate_mnemonics(types: list[TypeSpec]) -> None:
    """Fail at generation rather than at dialect registration."""
    counts = Counter(t.mnemonic for t in types)
    clashing = sorted(m for m, n in counts.items() if n > 1)
    if not clashing:
        return

    # Name the records behind each clash; the mnemonic alone does not say
    # which two declarations need disambiguating.
    detail = []
    for mnemonic in clashing:
        culprits = sorted(
            t.record
            for t in types
            if t.kind == "record" and t.mnemonic == mnemonic
        )
        detail.append(f"{mnemonic!r} <- {culprits}")
    raise DuplicateTypeMnemonicError(
        "type mnemonics must be unique within a dialect: " + "; ".join(detail)
    )


def plan_dialect(
    plans,
    options: TableGenOptions,
    *,
    records: set | None = None,
    enums: set | None = None,
    typedefs=None,
    record_objs: dict | None = None,
    unions: set | None = None,
) -> DialectModel:
    """The dialect ``plans`` implies under ``options``.

    ``plans`` is what :mod:`numbast.curate` produces, unchanged -- the
    declaration plus the exposed name. The remaining arguments are the header
    facts the type mapper needs to name and close over records.
    """
    mapper = TypeMapper(
        options.cpp_class_prefix,
        handle_prefix=options.handle_prefix,
        struct_prefixes=options.struct_prefixes,
        records=records,
        enums=enums,
        typedefs=typedefs,
        record_objs=record_objs,
        opaque_structs=options.opaque_structs,
    )

    # Ops first: resolving them is what populates the mapper.
    ops = [_resolve_op(plan, options, mapper, typedefs) for plan in plans]

    return DialectModel(
        dialect=_dialect_spec(options),
        types=collect_types(options, mapper, unions or frozenset()),
        ops=tuple(ops),
    )
