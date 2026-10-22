# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from pathlib import Path

import pytest

from ast_canopy import parse_declarations_from_source


@pytest.fixture(params=["numba-cuda", "numba-cuda-mlir"])
def generator(request):
    if request.param == "numba-cuda":
        runtime = pytest.importorskip("numba.cuda.types")
        if not hasattr(runtime, "bfloat16"):
            pytest.skip("requires the numba-cuda CUDA type registry")
        module = "numbast.tools.static_binding_generator"
    else:
        pytest.importorskip("numba_cuda_mlir")
        module = "numbast.experimental.mlir.tools.static_binding_generator"
    return request.param, import_module(module)


def test_python_generation_keeps_record_aliases_only(tmp_path, generator):
    backend, driver = generator
    source = tmp_path / "aliases.cuh"
    source.write_text(
        "typedef unsigned long long word_t;\n"
        "typedef word_t handle_t;\n"
        "struct Record { int value; };\n"
        "typedef Record record_t;\n"
    )
    config = driver.Config(
        {
            "Entry Point": str(source),
            "File List": [str(source)],
            "GPU Arch": ["sm_80"],
            "Backend": backend,
        }
    )
    driver.reset_renderer()
    path = driver._static_binding_generator(config, str(tmp_path))
    rendered = Path(path).read_text()
    compile(rendered, path, "exec")
    assert "record_t = Record" in rendered
    assert "word_t =" not in rendered
    assert "handle_t =" not in rendered


def test_record_alias_filter_preserves_chains_and_templates(
    tmp_path, generator
):
    _, driver = generator
    source = tmp_path / "aliases.cuh"
    source.write_text(
        "typedef int scalar_t;\n"
        "typedef int *pointer_t;\n"
        "struct Record {};\n"
        "typedef Record record_t;\n"
        "typedef record_t record_alias_t;\n"
        "template<class T> struct Box { T value; };\n"
        "typedef Box<int> box_t;\n"
    )
    declarations = parse_declarations_from_source(
        str(source), [str(source)], "sm_80"
    )
    assert driver._typedef_to_aliases(declarations.typedefs) == {
        "Record": ["record_t", "record_alias_t"],
        "Box<int>": ["box_t"],
    }
