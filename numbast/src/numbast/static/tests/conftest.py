# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import os
from typing import Any

from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import clear_function_apis_registry
from numbast.tools.static_binding_generator import (
    _static_binding_generator,
    Config,
)


@pytest.fixture(scope="session")
def data_folder():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    return lambda *file: os.path.join(current_directory, "data/", *file)


@pytest.fixture(scope="function")
def make_binding(tmpdir, data_folder):
    def _make_binding(
        header_name: str,
        types: dict[str, type],
        datamodels: dict[str, type],
        cc: str = "sm_80",
    ):
        clear_base_renderer_cache()
        clear_function_apis_registry()

        header_path = data_folder(header_name)
        cfg = Config.from_params(
            entry_point=header_path,
            retain_list=[header_path],
            gpu_arch=[cc],
            types=types,
            datamodels=datamodels,
            separate_registry=False,
        )
        _static_binding_generator(cfg, tmpdir)

        basename = header_name.split(".")[0]
        with open(tmpdir / f"{basename}.py") as f:
            src = f.read()

        bindings: dict[str, Any] = {}
        exec(src, bindings)

        return {
            "src": src,
            "bindings": bindings,
        }

    return _make_binding
