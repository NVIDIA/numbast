# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from numbast.tools import static_binding_generator as sbg


def test_cuda_toolkit_include_paths_are_forwarded(monkeypatch, tmp_path):
    header = tmp_path / "data.cuh"
    header.write_text("// header fixture\n", encoding="utf-8")
    cuda_include = tmp_path / "cuda" / "include"
    cuda_include.mkdir(parents=True)
    config = sbg.Config.from_params(
        entry_point=str(header),
        gpu_arch=["sm_80"],
        retain_list=[str(header)],
        types={},
        datamodels={},
        cuda_toolkit_include_paths=[str(cuda_include)],
    )
    captured = {}

    def parse_declarations(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            structs=[],
            functions=[],
            function_templates=[],
            enums=[],
            class_templates=[],
            typedefs=[],
        )

    monkeypatch.setattr(
        sbg, "parse_declarations_from_source", parse_declarations
    )

    sbg._static_binding_generator(config, str(tmp_path))

    assert captured["cudatoolkit_include_dirs"] == [str(cuda_include)]
