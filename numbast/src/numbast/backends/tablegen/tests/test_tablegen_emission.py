# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exactly what goes in, exactly what comes out.

``driverlike.h`` is the input and ``expected/`` is the output, compared
verbatim. Every construct in the header exists to produce something specific in
the expected files, so the two read as one document: a reviewer can see the
dialect this backend generates rather than infer it from assertions about
fragments.

The path is end to end and shares nothing with the Numba backends downstream of
parsing: header -> ast_canopy -> numbast.curate -> TableGen.

Regenerating
------------
``NUMBAST_REGEN_EXPECTED=1 pytest <this file>`` rewrites ``expected/`` from the
current renderers and passes. Read the resulting diff before committing it --
it is the entire behavioural change of whatever you just edited.
"""

import os

import pytest

from ast_canopy import parse_declarations_from_source

from numbast.backends.tablegen import (
    OutParamPolicy,
    TableGenFunctionsRenderer,
    TableGenOptions,
    TableGenTypesRenderer,
    header_facts,
    render_dialect_td,
)
from numbast.curate import plan_functions

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "driverlike.h")
EXPECTED = os.path.join(HERE, "expected")

REGEN = bool(os.environ.get("NUMBAST_REGEN_EXPECTED"))

#: The recipe the expected output was generated from. Kept here rather than in
#: a fixture so it reads as the YAML it stands in for.
OPTIONS = TableGenOptions(
    dialect_name="acme",
    cpp_class_prefix="Acme",
    handle_prefix="Acme",
    struct_prefixes=["Acme"],
    status_type="AcmeResult",
    out_param_policy=OutParamPolicy.C_OUT_PARAMS,
)


@pytest.fixture(scope="module")
def decls():
    return parse_declarations_from_source(HEADER, [HEADER], "sm_80")


@pytest.fixture(scope="module")
def generated(decls):
    """The dialect this backend emits for ``driverlike.h``, by file name."""
    facts = header_facts(decls)
    plans = plan_functions(
        decls.functions,
        header_path=HEADER,
        prefix_removal=["acme"],
        # A host C API has no __device__ functions; see TableGenOptions.
        skip_non_device=False,
    )

    # Ops first: rendering is what populates the mapper's type registry, which
    # the types renderer then declares.
    ops = TableGenFunctionsRenderer(
        plans,
        OPTIONS,
        records=facts["records"],
        enums=facts["enums"],
        typedefs=facts["typedefs"],
        record_objs=facts["record_objs"],
    )
    ops_td = ops.render_as_str()
    types_td = TableGenTypesRenderer(
        ops.mapper, OPTIONS, unions=facts["unions"]
    ).render_as_str()

    return {
        "AcmeDialect.td": render_dialect_td(OPTIONS),
        "AcmeTypes.td": types_td,
        "AcmeOps.td": ops_td,
    }


@pytest.mark.parametrize(
    "filename", ["AcmeDialect.td", "AcmeTypes.td", "AcmeOps.td"]
)
def test_emission_matches_expected(generated, filename):
    path = os.path.join(EXPECTED, filename)
    actual = generated[filename]

    if REGEN:
        os.makedirs(EXPECTED, exist_ok=True)
        with open(path, "w") as handle:
            handle.write(actual)
        return

    with open(path) as handle:
        assert actual == handle.read()


def test_curation_supplies_the_exposed_names(decls):
    """The exposed names in the expected output are not this backend's work.

    They come from the curation the Numba backends already use, which is the
    architectural claim of this backend: everything upstream of emission is
    shared.
    """
    plans = plan_functions(
        decls.functions,
        header_path=HEADER,
        prefix_removal=["acme"],
        skip_non_device=False,
    )
    by_symbol = {p.c_symbol: p.exposed_name for p in plans}
    assert by_symbol["acmeStreamCreate"] == "StreamCreate"
    assert by_symbol["acmeVersion"] == "Version"
