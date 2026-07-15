# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from numbast.experimental.mlir.tools import static_binding_generator as sbg


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
