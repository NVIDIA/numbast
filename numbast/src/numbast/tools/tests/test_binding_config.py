# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import pytest
import yaml

from numbast.tools.binding_config import CudaOxideConfig
from numbast.tools.static_binding_generator import Config


def _shared_config(tmp_path):
    header = tmp_path / "device_api.h"
    header.write_text(
        "__device__ int library_add(int, int);\n", encoding="utf-8"
    )
    return {
        "Name": "shared bindings",
        "Version": 1,
        "Entry Point": str(header),
        "GPU Arch": ["sm_90a"],
        "File List": [str(header)],
        "Exclude": {"Function": [], "Struct": []},
        "Predefined Macros": ["LIBRARY_DEVICE_API"],
        "API Prefix Removal": {"Function": "library_"},
        "Output Name": "bindings.py",
        "Types": {},
        "Data Models": {},
        "Function Argument Intents": {"library_add": {"0": "in"}},
        "CUDA Oxide": {
            "LTOIR Inputs": ["libdevice_api.ltoir"],
            "Output Name": "bindings.rs",
            "Manifest Name": "bindings.manifest.json",
        },
    }


def test_shared_document_builds_independent_backend_configs(tmp_path):
    raw_config = _shared_config(tmp_path)

    numba_config = Config(raw_config)
    rust_config = CudaOxideConfig(raw_config)

    assert numba_config.entry_point == rust_config.entry_point
    assert numba_config.gpu_arch == rust_config.gpu_arch == ["sm_90a"]
    assert (
        numba_config.api_prefix_removal
        == rust_config.api_prefix_removal
        == {"Function": ["library_"]}
    )
    assert numba_config.output_name == "bindings.py"
    assert rust_config.output_name == "bindings.rs"
    assert rust_config.manifest_name == "bindings.manifest.json"
    assert rust_config.parser_defines == [
        "LIBRARY_DEVICE_API",
        "__CUDA_ARCH__=900",
    ]


def test_shared_config_normalization_does_not_mutate_input(tmp_path):
    raw_config = _shared_config(tmp_path)
    original = copy.deepcopy(raw_config)

    numba_config = Config(raw_config)
    rust_config = CudaOxideConfig(raw_config)

    numba_config.gpu_arch.append("sm_100")
    numba_config.exclude_functions.append("other_function")
    rust_config.ltoir_inputs.append("other.ltoir")

    assert raw_config == original


def test_both_configs_load_same_yaml_with_numbast_tag(tmp_path):
    raw_config = _shared_config(tmp_path)
    header = raw_config["Entry Point"]
    config_path = tmp_path / "numbast.yaml"
    config_path.write_text(
        f"""
Entry Point: {header}
GPU Arch: [sm_90]
File List: [{header}]
Types: {{}}
Data Models: {{}}
Predefined Macros: [!numbast_join [LIBRARY_, DEVICE_API]]
CUDA Oxide:
  LTOIR Inputs: [libdevice_api.ltoir]
""",
        encoding="utf-8",
    )

    assert Config.from_yaml_path(config_path).predefined_macros == [
        "LIBRARY_DEVICE_API"
    ]
    assert CudaOxideConfig.from_yaml_path(config_path).predefined_macros == [
        "LIBRARY_DEVICE_API"
    ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("Output Name", "../bindings.rs"),
        ("Output Name", "generated/bindings.rs"),
        ("Manifest Name", "../bindings.json"),
        ("Manifest Name", "generated\\bindings.json"),
        ("Manifest Name", ".."),
    ],
)
def test_cuda_oxide_output_names_must_be_filenames(tmp_path, key, value):
    raw_config = _shared_config(tmp_path)
    raw_config["CUDA Oxide"][key] = value

    with pytest.raises(ValueError, match=rf"CUDA Oxide\.{key}.*filename"):
        CudaOxideConfig(raw_config)


def test_cuda_oxide_output_and_manifest_names_must_differ(tmp_path):
    raw_config = _shared_config(tmp_path)
    raw_config["CUDA Oxide"]["Manifest Name"] = "bindings.rs"

    with pytest.raises(ValueError, match="must differ"):
        CudaOxideConfig(raw_config)


def test_cuda_oxide_rejects_mismatched_cuda_arch_macro(tmp_path):
    raw_config = _shared_config(tmp_path)
    raw_config["Predefined Macros"].append("__CUDA_ARCH__=800")

    with pytest.raises(ValueError, match="expected __CUDA_ARCH__=900"):
        CudaOxideConfig(raw_config)


def test_cuda_oxide_rejects_unknown_nested_options(tmp_path):
    raw_config = _shared_config(tmp_path)
    raw_config["CUDA Oxide"]["Unknown"] = True

    with pytest.raises(ValueError, match='Unknown "CUDA Oxide".*Unknown'):
        CudaOxideConfig(raw_config)


def test_binding_config_loader_remains_safe(tmp_path):
    marker_path = tmp_path / "unsafe-loader-marker"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"!!python/object/apply:builtins.open\n- {marker_path}\n- w\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        CudaOxideConfig.from_yaml_path(config_path)

    assert not marker_path.exists()
