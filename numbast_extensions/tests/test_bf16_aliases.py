# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import runpy
from types import SimpleNamespace

import ast_canopy
import numbast
from numba import cuda
from numba.cuda import cuda_paths


def test_bf16_bindings_filter_unsupported_aliases(tmp_path, monkeypatch):
    source = tmp_path / "aliases.cuh"
    source.write_text(
        "typedef int scalar_t;\n"
        "typedef int *pointer_t;\n"
        "typedef enum { OK } status_t;\n"
        "struct Record {};\n"
        "typedef Record record_t;\n"
        "typedef record_t record_alias_t;\n"
        "union Storage { int value; float other; };\n"
        "typedef Storage storage_t;\n"
        "typedef storage_t storage_alias_t;\n"
    )
    declarations = ast_canopy.parse_declarations_from_source(
        str(source), [str(source)], "sm_80"
    )
    monkeypatch.setattr(
        ast_canopy, "parse_declarations_from_source", lambda *args: declarations
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(8, 0)),
    )
    monkeypatch.setattr(
        cuda_paths,
        "get_cuda_paths",
        lambda: {"include_dir": SimpleNamespace(info=str(tmp_path))},
    )

    class Record:
        pass

    def bind_structs(shim_writer, structs, type_specs, model_specs, aliases):
        assert aliases == {"Record": ["record_t", "record_alias_t"]}
        return [Record]

    # Exercise module initialization and exports without compiling GPU bindings.
    monkeypatch.setitem(numbast.__dict__, "bind_cxx_structs", bind_structs)
    monkeypatch.setitem(
        numbast.__dict__, "bind_cxx_functions", lambda *args, **kwargs: []
    )
    bindings = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "numbast_extensions/bf16/bf16_bindings.py"
        )
    )
    assert bindings["record_t"] is Record
    assert bindings["record_alias_t"] is Record
    assert set(bindings["__all__"]) == {
        "Record",
        "record_t",
        "record_alias_t",
        "get_shims",
    }
    assert all(name in bindings for name in bindings["__all__"])
