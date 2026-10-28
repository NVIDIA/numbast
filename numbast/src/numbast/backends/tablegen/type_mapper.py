# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""C/C++ type spellings -> MLIR type references.

Not an emitter: the shared mapper both ``.td`` renderers consume. The ops
renderer *references* types while rendering, which is what populates the
registries the types renderer then declares.

This is the piece with no counterpart in the existing backends. Numba emission
maps a spelling to a *Numba* type via a flat lookup (``to_numba_type_str``);
TableGen emission has to decide between a builtin (``I32``), a named opaque
handle type minted for this dialect (``Acme_StreamType``), a modelled struct, or
the dialect's catch-all pointer type.

Deciding that needs context the spelling alone does not carry -- which record
tags are complete, which names are enums, and what the dialect is called -- so
the mapper is stateful and also accumulates the handle types it mints.
"""

import re

__all__ = ["TypeMapper"]


class AmbiguousAnonymousRecordError(Exception):
    """Several unnamed nested records under one parent cannot be told apart.

    Raised rather than guessed at: the alternative is two distinct layouts
    sharing one dialect type, which miscompiles silently.
    """


# normalised spelling -> (byte width, unsigned)
_SCALARS: dict[str, tuple[int, bool]] = {
    "bool": (1, True),
    "char": (1, False),
    "signed char": (1, False),
    "unsigned char": (1, True),
    "short": (2, False),
    "unsigned short": (2, True),
    "int": (4, False),
    "unsigned int": (4, True),
    "long": (8, False),
    "unsigned long": (8, True),
    "long long": (8, False),
    "unsigned long long": (8, True),
    "size_t": (8, True),
    # Clang canonicalises some spellings with an explicit "int".
    "long int": (8, False),
    "unsigned long int": (8, True),
    "long long int": (8, False),
    "unsigned long long int": (8, True),
    "short int": (2, False),
    "unsigned short int": (2, True),
}

# Fixed-width typedefs clang does not always canonicalise to a primitive.
_ALIASES: dict[str, str] = {
    "int8_t": "signed char",
    "uint8_t": "unsigned char",
    "int16_t": "short",
    "uint16_t": "unsigned short",
    "int32_t": "int",
    "uint32_t": "unsigned int",
    "int64_t": "long long",
    "uint64_t": "unsigned long long",
    # Library-specific fixed-width typedefs clang does not expand. Without
    # these they look like unknown record names and get modelled as types.
    "cuuint32_t": "unsigned int",
    "cuuint64_t": "unsigned long long",
}


def _norm(spelling: str) -> str:
    """Strip qualifiers/whitespace noise so a spelling can be looked up."""
    s = spelling.replace("const", " ").replace("volatile", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return _ALIASES.get(s, s)


def _is_function_pointer(spelling: str) -> bool:
    return "(*" in spelling.replace(" ", "")


def _names_unnamed_record(spelling: str) -> bool:
    """Does this field's type refer to a record declared inline, with no tag?

    Clang spells those ``Parent::(unnamed union at file:line:col)``, and older
    releases ``(anonymous ...)``; both are accepted so the check does not turn
    on a clang version.
    """
    return "::" in spelling and (
        "(unnamed" in spelling or "(anonymous" in spelling
    )


def _is_pointer(spelling: str) -> bool:
    return spelling.rstrip().endswith("*") or spelling.rstrip().endswith("&")


def _pointee(spelling: str) -> str:
    """Strip exactly one level of pointer/reference."""
    s = spelling.rstrip()
    if s.endswith("*") or s.endswith("&"):
        s = s[:-1]
    return _norm(s)


class TypeMapper:
    """Map C spellings to MLIR type refs for one dialect.

    Parameters
    ----------
    cpp_class_prefix:
        TableGen record prefix, e.g. ``"Acme"`` -> ``Acme_PtrType``.
    handle_prefix:
        Prefix stripped when naming a minted handle type, e.g. ``"CU"`` so
        ``CUstream`` becomes ``stream`` / ``Acme_StreamType``.
    records:
        Names of *complete* structs/unions available in this translation unit.
    enums:
        Names of enum types; these lower to ``I32``.
    """

    def __init__(
        self,
        cpp_class_prefix: str,
        *,
        handle_prefix: str = "",
        struct_prefixes: list | None = None,
        records: set[str] | None = None,
        enums: set[str] | None = None,
        typedefs=None,
        record_objs: dict | None = None,
        opaque_structs=(),
    ):
        self.prefix = cpp_class_prefix
        self.handle_prefix = handle_prefix
        # Tried in order, longest first, so CUDA_ beats CU.
        self.struct_prefixes = list(struct_prefixes or [])
        self.records = records or set()
        self.enums = enums or set()
        # ast_canopy Record objects by name. Needed to take the *transitive*
        # closure: a modelled struct's record-typed fields are themselves types
        # the dialect must declare.
        self.record_objs = record_objs or {}
        self.opaque_structs = set(opaque_structs)
        self._modeled: set[str] = set()
        # TypedefFacts indexing the library's public names; without it handle
        # types get named after internal tags (Stream_st, not Stream).
        self.typedefs = typedefs
        # Minted handle types: C++ record name -> mnemonic.
        self.handles: dict[str, str] = {}
        # Referenced struct types: C name -> record name.
        self.structs: dict[str, str] = {}

    # -- naming -----------------------------------------------------------

    def _public(self, c_name: str) -> str:
        """Resolve an internal tag to its public typedef, when recovered."""
        return (
            self.typedefs.public(c_name)
            if self.typedefs is not None
            else c_name
        )

    @staticmethod
    def _camel(name: str) -> str:
        """``RESOURCE_DESC`` -> ``ResourceDesc``; ``stream`` -> ``Stream``."""
        name = re.sub(r"[^A-Za-z0-9_]", "", name)
        if not name:
            return ""
        if "_" in name or name.isupper():
            parts = [p for p in name.split("_") if p]
            return "".join(p[:1].upper() + p[1:].lower() for p in parts)
        return name[:1].upper() + name[1:]

    def _handle_record_name(self, c_name: str) -> str:
        """``CUstream_st`` -> ``Stream`` (with ``handle_prefix='CU'``)."""
        name = self._public(c_name)
        if self.handle_prefix and name.startswith(self.handle_prefix):
            name = name[len(self.handle_prefix) :]
        return self._camel(name)

    def _struct_record_name(self, c_name: str) -> str:
        """``CUDA_RESOURCE_DESC_st`` -> ``ResourceDesc``.

        Uses ``struct_prefixes`` rather than ``handle_prefix``: stripping only
        ``CU`` from ``CUDA_RESOURCE_DESC`` would leave ``DA_RESOURCE_DESC``.
        """
        name = self._public(c_name)
        if name.endswith("_st"):
            name = name[: -len("_st")]
        for prefix in self.struct_prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        return self._camel(name)

    def ptr_type(self) -> str:
        return f"{self.prefix}_PtrType"

    def handle_type(self, c_name: str) -> str:
        """Mint (or reuse) a named handle type for ``c_name``."""
        record = self._handle_record_name(c_name)
        if not record:
            return self.ptr_type()
        self.handles.setdefault(record, record.lower())
        return f"{self.prefix}_{record}Type"

    def struct_type(self, c_name: str) -> str:
        """Type ref for a named record, modelling it transitively if possible."""
        rec = self.record_objs.get(c_name)
        if rec is not None:
            return self.model_record(rec)

        # Anything carrying call syntax is a function type or function pointer,
        # not a record. Modelling it would mint a type named after the whole
        # signature (VoidVoidType, VoidCuasyncnotificationinfostVoidType, ...).
        if "(" in c_name or ")" in c_name:
            return self.ptr_type()

        record = self._struct_record_name(c_name)
        if not record:
            return self.ptr_type()
        self.structs.setdefault(c_name, record)
        return f"{self.prefix}_{record}Type"

    # -- transitive record closure ----------------------------------------

    def model_record(
        self, rec, parent: str = "", field: str = "", field_index: int = 0
    ) -> str:
        """Model a record and, recursively, every record its fields name.

        A dialect must declare not only the structs that appear in op
        signatures but everything reachable from them by field access.

        An anonymous nested record is named after the field that holds it,
        ``<Parent><Field>``. C also allows the *field* to be anonymous -- an
        inline ``union { ... };`` with no member name -- and then there is
        nothing to name it after, so its 0-based field index is used instead:
        ``<Parent>F<index>``, as the reference dialect spells it.

        The index is not decoration. Without it the name collapses to
        ``<Parent>_``, which camel-cases to a case-variant of the parent
        (``DevResource`` and ``Devresource``). TableGen keys on the C++ name
        and accepts the pair; the mnemonics then collide and the dialect
        aborts the process when it is registered.
        """
        if rec.name:
            c_name = rec.name
            record = self._struct_record_name(rec.name)
            bare = rec.name[:-3] if rec.name.endswith("_st") else rec.name
            if bare in self.opaque_structs or rec.name in self.opaque_structs:
                return self.ptr_type()
        else:
            suffix = field if field else f"F{field_index}"
            c_name = f"{parent}_{suffix}"
            record = self._camel(f"{parent}_{suffix}")

        if not record:
            return self.ptr_type()

        ref = f"{self.prefix}_{record}Type"
        if record in self._modeled:
            return ref  # already done; also guards recursive records
        self._modeled.add(record)
        self.structs.setdefault(c_name, record)

        fields = list(getattr(rec, "fields", ()) or ())
        # Which fields name an unnamed record, in order; a field's position in
        # this list is what picks its record out of the parent's unnamed ones.
        anon_fields = [
            i
            for i, f in enumerate(fields)
            if _names_unnamed_record(f.type_.name)
        ]
        for i, f in enumerate(fields):
            ordinal = anon_fields.index(i) if i in anon_fields else 0
            self._model_field(f, rec, record, i, ordinal)
        return ref

    def _model_field(
        self,
        f,
        parent_rec,
        parent_record: str,
        field_index: int,
        anon_ordinal: int,
    ) -> None:
        """Pull a field's type into the closure if it names a record."""
        spelling = f.type_.name.strip()

        if re.search(r"\[\d*\]$", spelling):
            return  # fixed/flexible array field: opaque floor

        if "::" in spelling:
            # Qualified name: an anonymous nested struct/union declared inline.
            # Clang spells these ``Parent::(anonymous union at file:line)``, so
            # this must be tested *before* any paren-based check.
            nested = self._resolve_nested(spelling, parent_rec, anon_ordinal)
            if nested is not None:
                self.model_record(
                    nested,
                    parent=parent_record,
                    field=f.name,
                    field_index=field_index,
                )
            return

        if _is_function_pointer(spelling) or "(" in spelling:
            return  # callback field: opaque pointer, nothing to declare

        norm = _norm(spelling)
        if norm in self.record_objs:
            self.model_record(
                self.record_objs[norm], parent=parent_record, field=f.name
            )
            return
        # Pointers, handles and scalars need no declaration beyond what map()
        # already mints.
        self.map(spelling)

    @staticmethod
    def _resolve_nested(spelling: str, parent_rec, anon_ordinal: int):
        """Find the nested record a qualified field type refers to.

        ``anon_ordinal`` counts how many *earlier* fields of this parent also
        named an unnamed record. A parent can hold several -- ``cuda.h``'s
        ``CUmemcpy3DOperand_st`` has a union whose ``ptr`` and ``array``
        fields are two different unnamed structs -- and ``Record`` exposes no
        source location to tell them apart, so position is the only signal
        there is. It is a sound one: clang reports unnamed nested records in
        declaration order, which is the order the fields naming them appear.
        """
        nested = getattr(parent_rec, "nested_records", ()) or ()

        # A qualified spelling can also name a *tagged* nested type, which is
        # unambiguous and takes precedence over any positional guess.
        tail = spelling.rsplit("::", 1)[-1]
        for r in nested:
            if r.name and r.name == tail:
                return r

        anon = [r for r in nested if not r.name]
        if not anon:
            return None
        if anon_ordinal < len(anon):
            return anon[anon_ordinal]
        raise AmbiguousAnonymousRecordError(
            f"{parent_rec.name or '<anonymous>'} names an unnamed record at "
            f"position {anon_ordinal} but reports only {len(anon)}: "
            f"{spelling!r}"
        )

    # -- mapping ----------------------------------------------------------

    def map(self, spelling: str) -> str:
        """MLIR type ref for a parameter or result spelling."""
        # Call syntax means a function pointer *or* a bare function type. The
        # latter has no ``(*`` so it slips past the pointer test, and would then
        # be modelled as a record named after the whole signature.
        if _is_function_pointer(spelling) or "(" in spelling:
            return self.ptr_type()

        if _is_pointer(spelling):
            pointee = _pointee(spelling)
            if pointee.endswith("*"):
                # pointer-to-pointer; out-handle cases are routed through the
                # intent plan instead, so this is a plain opaque pointer.
                return self.ptr_type()
            if pointee in self.records:
                return self.struct_type(pointee)
            if (
                pointee in _SCALARS
                or pointee in self.enums
                or pointee
                in (
                    "void",
                    "char",
                )
            ):
                return self.ptr_type()
            # A record tag we have no complete definition for: an opaque handle.
            return self.handle_type(pointee)

        norm = _norm(spelling)
        if norm == "void":
            return ""
        if norm == "float":
            return "F32"
        if norm == "double":
            return "F64"
        if norm in _SCALARS:
            width, _ = _SCALARS[norm]
            return f"I{width * 8}"
        if norm in self.enums:
            return "I32"
        if norm in self.records:
            return self.struct_type(norm)
        return self.ptr_type()

    def map_pointee(self, spelling: str) -> str:
        """MLIR type of what a pointer parameter points *at*.

        Used for out-parameters, where ``T *out`` is lifted to a result of
        type ``T``.
        """
        if not _is_pointer(spelling):
            return self.map(spelling)
        return self.map(_pointee(spelling))
