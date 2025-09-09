# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ast_canopy import parse_declarations_from_source


def test_function_with_attributes(data_folder):
    """Confirms that attribute names can be captured for functions."""
    srcstr = str(data_folder / "attributes.cu")
    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")

    ATTRS = {"maybe_unused", "noinline"}
    nfuncs = 0
    for nfuncs, fun in enumerate(iterfunctions(decls), 1):
        expected = set() if fun.name.endswith("noattr") else ATTRS
        assert fun.attributes == expected, (
            f"for item {fun.name!r} in {srcstr!r}"
        )
    assert nfuncs == 8  # Update as needed


def iterfunctions(decls):
    """Iterate the functions (and function templates) in a Declarations object."""
    for fun in decls.functions:
        yield fun

    for ft in decls.function_templates:
        yield ft.instantiate().function

    def _dostruct(struct):
        for fun in struct.methods:
            yield fun
        for fun in struct.templated_methods:
            yield fun.instantiate().function

    for struct in decls.structs:
        for fun in _dostruct(struct):
            yield fun

    for ct in decls.class_templates:
        for fun in _dostruct(ct.instantiate().record):
            yield fun
