# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: header -> ast_canopy -> numbast.curate -> TableGen.

Exercises the seam from a third backend's point of view: the plans come from
the same shared curation the Numba backends use, and only emission is new.
"""

import copy
import os
import re

import pytest

from ast_canopy import parse_declarations_from_source

from numbast.backends.tablegen import (
    OutParamPolicy,
    TableGenOptions,
    TableGenFunctionsRenderer,
    mnemonic_for,
)
from numbast.backends.tablegen.typedefs import recover_typedefs
from numbast.backends.tablegen.facts import header_facts
from numbast.curate import plan_functions

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "driverlike.h")


@pytest.fixture(scope="module")
def decls():
    return parse_declarations_from_source(HEADER, [HEADER], "sm_80")


@pytest.fixture(scope="module")
def rendered(decls):
    plans = plan_functions(
        decls.functions,
        header_path=HEADER,
        prefix_removal=["acme"],
        # A host C API has no __device__ functions; see TableGenOptions.
        skip_non_device=False,
    )
    cfg = TableGenOptions(
        dialect_name="acme",
        cpp_class_prefix="Acme",
        handle_prefix="Acme",
        status_type="AcmeResult",
        out_param_policy=OutParamPolicy.C_OUT_PARAMS,
    )
    renderer = TableGenFunctionsRenderer(
        plans,
        cfg,
        enums=header_facts(decls, HEADER)["enums"],
        typedefs=recover_typedefs(HEADER),
    )
    return renderer.render_as_str(), renderer


def test_anonymous_enum_is_an_integer_operand(rendered):
    """An untagged ``typedef enum {...} Name;`` must not become a pointer.

    ast_canopy reports no enum for it and there is no tag to recover it from,
    so it has to be found by its public name.
    """
    text, _ = rendered
    block = re.search(
        r"def Acme_GetResourceOp\b.*?let arguments = \(ins([^)]*)\)",
        text,
        re.S,
    )
    assert block and "I32:$type_" in block.group(1), block and block.group(1)


@pytest.mark.parametrize(
    "exposed,expected",
    [
        ("streamCreate", "stream.create"),
        ("streamWaitEvent", "stream.wait.event"),
        ("deviceGetCount", "device.get.count"),
        ("version", "version"),
    ],
)
def test_mnemonic_splitting(exposed, expected):
    assert mnemonic_for(exposed) == expected


def test_curation_supplies_exposed_names(decls):
    """The exposed name comes from shared curation, not from this backend."""
    plans = plan_functions(
        decls.functions,
        header_path=HEADER,
        prefix_removal=["acme"],
        # A host C API has no __device__ functions; see TableGenOptions.
        skip_non_device=False,
    )
    by_symbol = {p.c_symbol: p.exposed_name for p in plans}
    assert by_symbol["acmeStreamCreate"] == "StreamCreate"
    assert by_symbol["acmeVersion"] == "Version"


def test_out_param_is_lifted_to_a_result(rendered):
    """``AcmeStream *phStream`` leaves the operand list and becomes a result."""
    text, _ = rendered
    assert "def Acme_StreamCreateOp" in text
    block = text.split("def Acme_StreamCreateOp")[1].split("def ")[0]
    assert "$Flags" in block
    assert "I32:$Flags" in block
    # the out-param is a result, typed as the handle it points at
    assert "Acme_StreamType:$phStream" in block
    # and is gone from the operands
    assert "(ins I32:$Flags)" in block


def test_status_return_trails_the_results(rendered):
    text, _ = rendered
    block = text.split("def Acme_StreamCreateOp")[1].split("def ")[0]
    results = block.split("(outs")[1].split(")")[0]
    assert results.strip().endswith("Acme_ResultType:$result")


def test_handle_operands_are_named_types(rendered):
    text, _ = rendered
    block = text.split("def Acme_StreamWaitEventOp")[1].split("def ")[0]
    assert "Acme_StreamType:$hStream" in block
    assert "Acme_EventType:$hEvent" in block
    # no out-params here, so status is the only result
    assert "(outs Acme_ResultType:$result)" in block


def test_scalar_out_param(rendered):
    text, _ = rendered
    block = text.split("def Acme_DeviceGetCountOp")[1].split("def ")[0]
    assert "I32:$count" in block
    assert "(ins)" in block


def test_non_status_return_becomes_a_leading_result(rendered):
    """``int acmeVersion(void)`` has a real return value, not a status."""
    text, _ = rendered
    block = text.split("def Acme_VersionOp")[1].split("def ")[0]
    results = block.split("(outs")[1].split(")")[0]
    # "result" is reserved by mlir-tblgen, so a real return value
    # falls through to the next free candidate.
    assert results.strip().startswith("I32:$ret")


def test_non_status_return_gets_no_phantom_status(rendered):
    """A function that does not return the status type must not gain one."""
    text, _ = rendered
    block = text.split("def Acme_VersionOp")[1].split("def ")[0]
    assert "Acme_ResultType" not in block
    assert "(outs I32:$ret)" in block


def test_shim_symbol(rendered):
    text, _ = rendered
    assert 'return std::string("_acmeStreamCreate")' in text


def test_handle_types_are_minted_for_declaration(rendered):
    """The mapper accumulates the handle types the dialect must also define."""
    _, renderer = rendered
    assert renderer.mapper.handles == {"Stream": "stream", "Event": "event"}


# -- type / dialect definitions ---------------------------------------------


@pytest.fixture(scope="module")
def types_td(rendered):
    from numbast.backends.tablegen import TableGenTypesRenderer

    _, renderer = rendered
    cfg = TableGenOptions(
        dialect_name="acme",
        cpp_class_prefix="Acme",
        handle_prefix="Acme",
        status_type="AcmeResult",
    )
    return TableGenTypesRenderer(renderer.mapper, cfg).render_as_str()


def test_types_file_is_guarded_and_defines_the_base_class(types_td):
    assert "#ifndef ACME_TYPES_TD" in types_td
    assert "#endif // ACME_TYPES_TD" in types_td
    assert "class Acme_Type<string name, string typeMnemonic>" in types_td
    assert "TypeDef<Acme_Dialect, name>" in types_td


def test_status_and_opaque_pointer_types_are_declared(types_td):
    assert 'def Acme_ResultType : Acme_Type<"Result", "result">' in types_td
    assert "AcmeResult status code" in types_td
    assert 'def Acme_PtrType : Acme_Type<"Ptr", "ptr">' in types_td


def test_every_referenced_handle_gets_a_typedef(types_td):
    """Ops referenced Acme_StreamType/Acme_EventType, so both must be declared."""
    assert 'def Acme_StreamType : Acme_Type<"Stream", "stream">' in types_td
    assert 'def Acme_EventType : Acme_Type<"Event", "event">' in types_td
    assert "opaque acme stream handle" in types_td


def test_records_differing_only_in_case_are_refused(rendered):
    """Two records whose names differ only in case share one mnemonic.

    A struct's mnemonic is its record name lowered, so ``DevResource`` and
    ``Devresource`` -- a real record and the anonymous one nested inside it --
    both ask for ``devresource``. TableGen keys on the C++ name and accepts
    it, the library links, and the process then dies at registration with
    ``Dialect Type with name ... is already registered``. Generation is the
    last place this can be reported against the declarations that caused it.
    """
    from numbast.backends.tablegen import TableGenTypesRenderer
    from numbast.backends.tablegen.types_td import DuplicateTypeMnemonicError

    _, renderer = rendered
    mapper = copy.deepcopy(renderer.mapper)
    mapper.structs["AcmeDevResource_st"] = "DevResource"
    mapper.structs["AcmeDevResource_anon"] = "Devresource"

    cfg = TableGenOptions(dialect_name="acme", cpp_class_prefix="Acme")
    with pytest.raises(DuplicateTypeMnemonicError) as excinfo:
        TableGenTypesRenderer(mapper, cfg).render_as_str()

    # The message has to name both culprits, or it sends the reader hunting
    # through a hundred-odd TypeDefs for which two collided.
    message = str(excinfo.value)
    assert "'devresource'" in message
    assert "DevResource" in message and "Devresource" in message


def test_distinct_record_names_are_accepted(rendered):
    """The control for the test above: one case change makes it legal."""
    from numbast.backends.tablegen import TableGenTypesRenderer

    _, renderer = rendered
    mapper = copy.deepcopy(renderer.mapper)
    mapper.structs["AcmeDevResource_st"] = "DevResource"
    mapper.structs["AcmeDevResource_anon"] = "DevResourceAnon"

    cfg = TableGenOptions(dialect_name="acme", cpp_class_prefix="Acme")
    td = TableGenTypesRenderer(mapper, cfg).render_as_str()
    assert '"devresource"' in td
    assert '"devresourceanon"' in td


def test_dialect_td(rendered):
    from numbast.backends.tablegen import render_dialect_td

    cfg = TableGenOptions(dialect_name="acme", cpp_class_prefix="Acme")
    td = render_dialect_td(cfg)
    assert "def Acme_Dialect : Dialect" in td
    assert 'let name = "acme";' in td
    assert 'let cppNamespace = "::mlir::acme";' in td
    assert "let useDefaultTypePrinterParser = 1;" in td


class _Type:
    def __init__(self, name):
        self.name = name
        self.unqualified_non_ref_type_name = name

    def is_left_reference(self):
        return False

    def is_right_reference(self):
        return False


class _Field:
    def __init__(self, name, type_name):
        self.name = name
        self.type_ = _Type(type_name)


class _Rec:
    def __init__(self, name, fields=(), nested=()):
        self.name = name
        self.fields = list(fields)
        self.nested_records = list(nested)
        self.is_union = False


def test_transitive_closure_models_nested_records():
    """A modelled struct's record-typed fields must also become types."""
    from numbast.backends.tablegen.types import TypeMapper

    inner = _Rec("AcmeExtent", [_Field("w", "unsigned int")])
    outer = _Rec("AcmeDesc", [_Field("extent", "AcmeExtent")])

    m = TypeMapper(
        "Acme",
        records={"AcmeDesc", "AcmeExtent"},
        record_objs={"AcmeDesc": outer, "AcmeExtent": inner},
    )
    assert m.struct_type("AcmeDesc") == "Acme_AcmeDescType"
    # the nested record was pulled in even though no op mentions it
    assert "AcmeExtent" in m.structs


def test_an_unnamed_field_is_named_by_its_index():
    """C lets the *field* be anonymous too, and then there is nothing to
    name the record after.

    ``union { ... };`` with no member name leaves the field name empty, so
    ``<Parent>_<Field>`` degenerates to ``<Parent>_``. That camel-cases to a
    case-variant of the parent -- ``AcmeDevResource`` and
    ``Acmedevresource`` -- which TableGen accepts and the dialect then
    rejects at registration, because both lower to one mnemonic. The field
    index is what keeps the two apart.
    """
    from numbast.backends.tablegen.types import TypeMapper

    anon = _Rec("", [_Field("sm", "int")])
    parent = _Rec(
        "AcmeDevResource_st",
        [
            _Field("type", "int"),
            _Field("_padding", "char[8]"),
            _Field("", "AcmeDevResource_st::(unnamed union at h.h:9:5)"),
        ],
        nested=[anon],
    )

    m = TypeMapper(
        "Acme",
        records={"AcmeDevResource_st"},
        record_objs={"AcmeDevResource_st": parent},
    )
    m.struct_type("AcmeDevResource_st")

    names = sorted(m.structs.values())
    assert len(names) == 2, names
    parent_name, anon_name = sorted(names, key=len)
    # Compared case-insensitively because splitting on ``_`` lowers the tail,
    # which is how the reference spells these too (``DevresourceF2``). The
    # casing is finding 22's separate argument; what matters here is that the
    # index reached the name and the two cannot collapse to one mnemonic.
    assert anon_name.lower() == parent_name.lower() + "f2", names
    assert len({n.lower() for n in names}) == 2, names


def test_two_unnamed_records_under_one_parent_stay_distinct():
    """Position is the only thing telling sibling unnamed records apart.

    ``Record`` carries no source location, so if the resolver cannot count
    fields it hands back the same record for both -- one layout modelled
    twice and the other not at all. Each sibling here reaches a *different*
    further record, so that failure shows up as a missing type rather than
    merely a duplicated name.
    """
    from numbast.backends.tablegen.types import TypeMapper

    location = _Rec("AcmeLocation", [_Field("id", "int")])
    offset = _Rec("AcmeOffset", [_Field("x", "int")])
    first = _Rec("", [_Field("locHint", "AcmeLocation")])
    second = _Rec("", [_Field("off", "AcmeOffset")])

    parent = _Rec(
        "AcmeOperand_st",
        [
            _Field("ptr", "AcmeOperand_st::(unnamed struct at h.h:3:5)"),
            _Field("array", "AcmeOperand_st::(unnamed struct at h.h:7:5)"),
        ],
        nested=[first, second],
    )

    m = TypeMapper(
        "Acme",
        records={"AcmeOperand_st", "AcmeLocation", "AcmeOffset"},
        record_objs={
            "AcmeOperand_st": parent,
            "AcmeLocation": location,
            "AcmeOffset": offset,
        },
    )
    m.struct_type("AcmeOperand_st")

    # Both siblings were followed, so both of their onward records appear.
    assert "AcmeLocation" in m.structs, m.structs
    assert "AcmeOffset" in m.structs, m.structs
    assert len(set(m.structs.values())) == len(m.structs), m.structs


def test_function_pointer_fields_do_not_become_types():
    """A callback field is an opaque pointer, not a record named after its signature."""
    from numbast.backends.tablegen.types import TypeMapper

    m = TypeMapper("Acme")
    assert m.map("void (*)(int, void *)") == "Acme_PtrType"
    # a bare function type has no "(*" and must still not be modelled
    assert m.map("void (int, void *)") == "Acme_PtrType"
    assert m.structs == {}
