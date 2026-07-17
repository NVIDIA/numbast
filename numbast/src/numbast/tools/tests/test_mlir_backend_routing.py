# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from numbast.tools import static_binding_generator as sbg


def _minimal_config(tmp_path):
    header = tmp_path / "input.cuh"
    header.write_text("__device__ int f();\n", encoding="utf-8")
    return {
        "Entry Point": str(header),
        "GPU Arch": ["sm_80"],
        "File List": [str(header)],
        "Types": {},
        "Data Models": {},
    }


def test_cfg_path_uses_mlir_backend(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump({"MLIR Backend": True}), encoding="utf-8")

    assert sbg._cfg_path_uses_mlir_backend(cfg_path)


def test_cfg_path_supports_numbast_join_tag(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        'MLIR Backend: !numbast_join ["tr", "ue"]\n',
        encoding="utf-8",
    )

    assert sbg._cfg_path_uses_mlir_backend(cfg_path)


@pytest.mark.parametrize(
    "load_config",
    [sbg._cfg_path_uses_mlir_backend, sbg.Config.from_yaml_path],
)
def test_config_load_rejects_python_object_tags(tmp_path, load_config):
    marker_path = tmp_path / "unsafe-loader-marker"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"!!python/object/apply:builtins.open\n- {marker_path}\n- w\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        load_config(cfg_path)

    assert not marker_path.exists()


def test_static_generator_dispatches_mlir_backend(monkeypatch, tmp_path):
    class DummyConfig:
        mlir_backend = True

    calls = {}

    def fake_mlir_config(config):
        calls["source_config"] = config
        return "mlir-config"

    def fake_mlir_generator(config, output_dir, **kwargs):
        calls["config"] = config
        calls["output_dir"] = output_dir
        calls["kwargs"] = kwargs
        return "generated.py"

    monkeypatch.setattr(sbg, "_mlir_config_from_config", fake_mlir_config)
    monkeypatch.setattr(
        sbg, "_run_mlir_static_binding_generator", fake_mlir_generator
    )

    output = sbg._static_binding_generator(
        DummyConfig(),
        str(tmp_path),
        log_generates=True,
        cfg_file_path="config.yaml",
        sbg_params={"x": "y"},
        bypass_parse_error=True,
    )

    assert output == "generated.py"
    assert calls["source_config"].mlir_backend
    assert calls["config"] == "mlir-config"
    assert calls["output_dir"] == str(tmp_path)
    assert calls["kwargs"] == {
        "log_generates": True,
        "cfg_file_path": "config.yaml",
        "sbg_params": {"x": "y"},
        "bypass_parse_error": True,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("Module Link Variables Used", ["x"]),
    ],
)
def test_mlir_only_config_requires_mlir_backend(tmp_path, key, value):
    config = _minimal_config(tmp_path)
    config[key] = value

    with pytest.raises(ValueError, match=f"{key}.*MLIR Backend: true"):
        sbg.Config(config)


def test_mlir_only_config_allowed_with_mlir_backend(tmp_path):
    config = _minimal_config(tmp_path)
    config.update(
        {
            "MLIR Backend": True,
            "Module Link Variables Used": ["x"],
        }
    )

    assert sbg.Config(config).mlir_backend
