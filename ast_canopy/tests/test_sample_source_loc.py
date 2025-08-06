# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ast_canopy import parse_declarations_from_source


def test_source_location(data_folder):
    srcstr = str(data_folder / "sample_source_loc.cu")

    decls = parse_declarations_from_source(srcstr, [srcstr], "sm_80")

    assert len(decls.functions) == 1
    assert len(decls.structs) == 1
    assert len(decls.class_templates) == 1
    assert len(decls.function_templates) == 1

    foo = decls.functions[0]
    assert "sample_source_loc.cu" in foo.source_location.file_name
    assert foo.source_location.line == 6
    assert foo.source_location.column == 33
    assert foo.source_location.is_valid

    bar = decls.structs[0]
    assert "sample_source_loc.cu" in bar.source_location.file_name
    assert bar.source_location.line == 8
    assert bar.source_location.column == 8
    assert bar.source_location.is_valid

    assert len(bar.methods) == 1
    barctor = bar.methods[0]
    assert "sample_source_loc.cu" in barctor.source_location.file_name
    assert barctor.source_location.line == 9
    assert barctor.source_location.column == 5
    assert barctor.source_location.is_valid

    # The template has the same source location as the function
    baz = decls.function_templates[0]
    assert "sample_source_loc.cu" in baz.source_location.file_name
    assert baz.source_location.line == 13
    assert baz.source_location.column == 33
    assert baz.source_location.is_valid

    assert "sample_source_loc.cu" in baz.function.source_location.file_name
    assert baz.function.source_location.line == 13
    assert baz.function.source_location.column == 33
    assert baz.function.source_location.is_valid

    # The template has the same source location as the class
    bax = decls.class_templates[0]
    assert "sample_source_loc.cu" in bax.source_location.file_name
    assert bax.source_location.line == 16
    assert bax.source_location.column == 8
    assert bax.source_location.is_valid

    assert "sample_source_loc.cu" in bax.record.source_location.file_name
    assert bax.record.source_location.line == 16
    assert bax.record.source_location.column == 8
    assert bax.record.source_location.is_valid

    watermelon = decls.enums[0]
    assert "sample_source_loc.cu" in watermelon.source_location.file_name
    assert watermelon.source_location.line == 18
    assert watermelon.source_location.column == 12
    assert watermelon.source_location.is_valid

    suika = decls.typedefs[0]
    assert "sample_source_loc.cu" in suika.source_location.file_name
    assert suika.source_location.line == 20
    assert suika.source_location.column == 20
    assert suika.source_location.is_valid
