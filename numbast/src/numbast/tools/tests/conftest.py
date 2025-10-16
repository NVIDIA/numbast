# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import warnings

import pytest

from numba import cuda

from jinja2 import Environment, FileSystemLoader

from click.testing import CliRunner

from numbast.tools.static_binding_generator import static_binding_generator


@pytest.fixture
def run_in_isolated_folder(tmpdir):
    # Helper to simulate a production environment where configurations are used
    # Tmp Folder structure:
    # - /
    # - config/
    #   - <config_name>.yml
    # - output/
    #   - <header_name>.py
    #
    # Test folder structure:
    # - .
    # - config
    #   - <template_a>.yml.j2
    #   - <template_b>.yml.j2
    # - <header_a>.cuh
    # - <header_b>.cuh
    # - test_a.py
    # - test_b.py
    def _run(
        cfg_template,
        header,
        params,
        output_name=None,
        ruff_format=False,
        load_symbols=False,
        show_binding=False,
        bypass_parse_error=False,
    ):
        root = tmpdir
        config_folder = root.mkdir("config")
        output_folder = root.mkdir("output")
        here = os.path.dirname(os.path.abspath(__file__))

        src_data = os.path.join(here, header)
        target_data = os.path.join(output_folder, header)
        config_name = cfg_template.replace(".j2", "")
        config_path = os.path.join(config_folder, config_name)
        shutil.copy(src_data, target_data)

        params["data"] = target_data
        if output_name is not None:
            params["output_name"] = output_name

        env = Environment(loader=FileSystemLoader(here))
        template = env.get_template(os.path.join("config/", cfg_template))
        config = template.render(params)

        with open(config_path, "w") as f:
            f.write(config)

        runner = CliRunner(catch_exceptions=False)

        with warnings.catch_warnings(record=True) as w:
            result = runner.invoke(
                static_binding_generator,
                [
                    "--cfg-path",
                    config_path,
                    "--output-dir",
                    output_folder,
                    "-fmt",
                    "true" if ruff_format else "false",
                    "-noraise",
                    "true" if bypass_parse_error else "false",
                ],
            )

        assert result.exit_code == 0, result.stdout

        if output_name is None:
            output_name = header.split(".")[0] + ".py"
        binding_path = os.path.join(output_folder, output_name)

        with open(binding_path) as f:
            binding = f.read()

        symbols = {}
        if load_symbols:
            exec(binding, symbols)

        if show_binding:
            print(binding)

        return {
            "result": result,
            "output_folder": output_folder,
            "binding_path": binding_path,
            "binding": binding,
            "symbols": symbols,
            "warnings": w,
        }

    return _run


@pytest.fixture
def arch_str():
    arch = cuda.get_current_device().compute_capability
    return f"sm_{arch[0]}{arch[1]}"
