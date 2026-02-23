# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import ast_canopy.api as api


def test_parse_declarations_falls_back_to_clangpp(monkeypatch, tmp_path):
    source = tmp_path / "sample.cu"
    source.write_text('extern "C" __global__ void kernel() {}\n')

    captured: dict[str, list[str]] = {}
    empty_decls = SimpleNamespace(
        records=[],
        functions=[],
        function_templates=[],
        class_templates=[],
        class_template_specializations=[],
        typedefs=[],
        enums=[],
    )

    monkeypatch.setattr(api, "check_clang_binary", lambda: None)
    monkeypatch.setattr(
        api, "get_clang_resource_dir", lambda clang_binary: "/lib/clang/20/"
    )
    monkeypatch.setattr(
        api, "get_default_compiler_search_paths", lambda clang_binary: []
    )
    monkeypatch.setattr(api, "get_cuda_include_dir_for_clang", lambda: {})
    monkeypatch.setattr(api, "get_cuda_path_for_clang", lambda: "/tmp/cuda")

    def _fake_parse(command_line_options, files_to_retain, bypass_parse_error):
        captured["command_line_options"] = command_line_options
        return empty_decls

    monkeypatch.setattr(
        api.bindings, "parse_declarations_from_command_line", _fake_parse
    )

    api.parse_declarations_from_source(
        str(source), [str(source)], "sm_80", cudatoolkit_include_dirs=[]
    )

    assert captured["command_line_options"][0] == "clang++"


def test_value_from_constexpr_uses_detected_clang_binary(monkeypatch):
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        api, "check_clang_binary", lambda: "/usr/bin/custom-clang++"
    )
    monkeypatch.setattr(
        api, "get_clang_resource_dir", lambda clang_binary: "/lib/clang/20/"
    )
    monkeypatch.setattr(
        api, "get_default_compiler_search_paths", lambda clang_binary: []
    )
    monkeypatch.setattr(api, "get_cuda_include_dir_for_clang", lambda: {})
    monkeypatch.setattr(api, "get_cuda_path_for_clang", lambda: "/tmp/cuda")
    monkeypatch.setattr(api, "_get_shim_include_dir", lambda: "/tmp/shim")

    def _fake_value_from_constexpr(command_line_options, vardecl_name):
        captured["command_line_options"] = command_line_options
        return None

    monkeypatch.setattr(
        api.bindings, "value_from_constexpr_vardecl", _fake_value_from_constexpr
    )

    result = api.value_from_constexpr_vardecl(
        "constexpr int k = 1;", "k", "sm_80"
    )

    assert result is None
    assert captured["command_line_options"][0] == "/usr/bin/custom-clang++"
