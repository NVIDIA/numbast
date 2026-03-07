# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import os
from typing import Any

from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import clear_function_apis_registry
from numbast.static.function_template import clear_function_template_registry
from numbast.static.class_template import clear_class_template_cache
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
    """
    Return a factory that generates static Python bindings for a C/C++ header into the given temporary directory.

    Parameters:
        tmpdir (os.PathLike | py.path.local): Directory where generated binding files will be written.
        data_folder (Callable[[str], str]): Function that maps a header filename to its full path within the test data folder.

    Returns:
        Callable[[str, dict[str, type], dict[str, type], str, dict | None], dict]:
            A factory function with signature
                (header_name, types, datamodels, cc="sm_80", function_argument_intents=None) -> dict
            The factory generates bindings for `header_name` and returns a dictionary with:
                - "src": the generated Python source code as a string.
                - "bindings": a dict populated by executing the generated source.
    """

    def _make_binding(
        header_name: str,
        types: dict[str, type],
        datamodels: dict[str, type],
        cc: str = "sm_80",
        function_argument_intents: dict | None = None,
    ):
        clear_base_renderer_cache()
        clear_function_apis_registry()
        clear_function_template_registry()
        clear_class_template_cache()

        header_path = data_folder(header_name)
        cfg = Config.from_params(
            entry_point=header_path,
            retain_list=[header_path],
            gpu_arch=[cc],
            types=types,
            datamodels=datamodels,
            separate_registry=False,
        )
        cfg.function_argument_intents = function_argument_intents or {}
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
