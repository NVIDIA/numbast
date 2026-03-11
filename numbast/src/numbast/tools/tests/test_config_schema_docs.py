# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import yaml

from numbast.tools.static_binding_generator import Config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _schema_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "static_binding_generator.schema.yaml"
    )


def test_static_binding_schema_has_expected_keys():
    schema = yaml.safe_load(_schema_path().read_text(encoding="utf-8"))

    required = set(schema["required"])
    assert required == {"Entry Point", "GPU Arch", "File List"}

    properties = schema["properties"]
    expected_keys = {
        "Entry Point",
        "GPU Arch",
        "File List",
        "Types",
        "Data Models",
        "Exclude",
        "Clang Include Paths",
        "Additional Import",
        "Shim Include Override",
        "Predefined Macros",
        "Output Name",
        "Cooperative Launch Required Functions Regex",
        "API Prefix Removal",
        "Module Callbacks",
        "Skip Prefix",
        "Use Separate Registry",
        "Function Argument Intents",
    }
    assert expected_keys.issubset(set(properties))

    gpu_arch = properties["GPU Arch"]
    assert gpu_arch["type"] == "array"
    assert gpu_arch["maxItems"] == 1
    assert gpu_arch["items"]["pattern"] == "^sm_[0-9]+$"


def test_generate_schema_reference_from_yaml_schema(tmp_path):
    module_path = (
        _repo_root()
        / "docs"
        / "source"
        / "_ext"
        / "static_binding_schema_doc.py"
    )
    spec = importlib.util.spec_from_file_location(
        "static_binding_schema_doc", module_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output_path = tmp_path / "static_binding_schema_reference.rst"
    module.generate_static_binding_schema_reference(
        schema_path=_schema_path(),
        output_path=output_path,
        schema_repo_path="numbast/src/numbast/tools/static_binding_generator.schema.yaml",
    )

    rendered = output_path.read_text(encoding="utf-8")
    assert "Required keys" in rendered
    assert "Optional keys" in rendered
    assert "``Entry Point``" in rendered
    assert "``Use Separate Registry``" in rendered
    assert "Raw schema" in rendered
    assert "$schema: " in rendered


def test_from_params_sets_use_separate_registry(tmp_path):
    header = tmp_path / "data.cuh"
    header.write_text("// header fixture\n", encoding="utf-8")

    config = Config.from_params(
        entry_point=str(header),
        gpu_arch=["sm_80"],
        retain_list=[str(header)],
        types={},
        datamodels={},
        separate_registry=True,
    )

    assert config.separate_registry is True
