# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
from pathlib import Path

from ast_canopy import parse_declarations_from_source
from ast_canopy.decl import EvaluatedConstDecl


def test_global_scalar_constants_are_evaluated():
    source = str(
        Path(__file__).with_name("data") / "sample_global_constants.cu"
    )
    declarations = parse_declarations_from_source(source, [source], "sm_80")
    for constants in (
        declarations.global_constants,
        pickle.loads(pickle.dumps(declarations.global_constants)),
    ):
        by_name = {item.name: item for item in constants}
        assert set(by_name) == {
            "BASE",
            "GLOBAL_COUNT",
            "MAX_U64",
            "ENABLED",
            "DISABLED",
            "RATIO",
            "SCALE",
            "DEFAULT_HANDLE",
            "DEFAULT_STATUS",
            "NAMESPACE_VALUE",
        }
        assert all(isinstance(item, EvaluatedConstDecl) for item in constants)
        assert {item.name for item in constants if item.is_constexpr} == {
            "BASE",
            "ENABLED",
            "RATIO",
            "DEFAULT_HANDLE",
            "NAMESPACE_VALUE",
        }
        assert by_name["GLOBAL_COUNT"].is_constexpr is False
        assert by_name["GLOBAL_COUNT"].value == 42
        assert by_name["MAX_U64"].value == 2**64 - 1
        assert by_name["ENABLED"].value is True
        assert by_name["DISABLED"].value is False
        assert by_name["RATIO"].value == 1.2345678901234567
        assert by_name["SCALE"].value == 1.25
        assert by_name["DEFAULT_HANDLE"].value == -1
        assert by_name["DEFAULT_STATUS"].value == 2
        assert (
            by_name["NAMESPACE_VALUE"].qual_name == "library::NAMESPACE_VALUE"
        )


def test_constants_respect_retained_headers(tmp_path):
    included = tmp_path / "included.h"
    included.write_text("constexpr int INCLUDED = 5;\n")
    source = tmp_path / "entry.cu"
    source.write_text(
        '#include "included.h"\nconstexpr int LOCAL = INCLUDED * 2;\n'
    )
    declarations = parse_declarations_from_source(
        str(source), [str(source)], "sm_80"
    )
    assert [
        (item.name, item.value) for item in declarations.global_constants
    ] == [("LOCAL", 10)]
