# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from click.testing import CliRunner

import numpy as np
from numba import cuda
import pytest

from numbast.static.renderer import clear_base_renderer_cache
from numbast.static.function import clear_function_apis_registry
from numbast.tools.static_binding_generator import static_binding_generator


@pytest.fixture
def kernel():
    def _lazy_kernel(globals):
        Foo = globals["Foo"]
        add = globals["add"]

        @cuda.jit
        def kernel(arr):
            foo = Foo()
            arr[0] = foo.x
            arr[1] = add(1, 2)

        arr = np.array([42, 0])
        kernel[1, 1](arr)
        assert arr[0] == 0
        assert arr[1] == 3

    return _lazy_kernel


@pytest.mark.parametrize(
    "inputs",
    [
        ["/tmp/non_existing.cuh"],
        [
            os.path.join(os.path.dirname(__file__), "data.cuh"),
            "--output-dir",
            "/tmp/non_existing/",
        ],
        [
            os.path.join(os.path.dirname(__file__), "data.cuh"),
            "--output-dir",
            "/tmp/",
            "--types",
            "invalid_type_string",
        ],
        [
            os.path.join(os.path.dirname(__file__), "data.cuh"),
            "--output-dir",
            "/tmp/",
            "--types",
            '{"Foo":"Type"}',
            "--datamodels",
            "invalid_datamodel_string",
        ],
    ],
)
def test_invalid_input_header(inputs):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    runner = CliRunner()
    with pytest.raises(Exception):
        result = runner.invoke(static_binding_generator, inputs)
        assert result.exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--entry-point", "data.cuh"],
        ["--types", '{"Foo":"Type"}'],
        ["--datamodels", '{"Foo": "StructModel"}'],
    ],
)
def test_cli_yml_invalid_inputs(tmpdir, args):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude: []
Types:
    Foo: Type
Data Models:
    Foo: StructModel
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ]
        + args,
    )

    with pytest.raises(Exception):
        assert result.exit_code == 0


@pytest.mark.skip("TODO: A C++ error is thrown.")
def test_simple_cli_empty_retain(tmpdir):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub3")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()

    result = runner.invoke(
        static_binding_generator,
        [data, "--output-dir", subdir, "--retain", ""],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    # Both are not in the retain list
    assert "Foo" not in globals
    assert "add" not in globals


def test_cli_yml_inputs_full_spec(tmpdir, kernel):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude: {{}}
Types:
    Foo: Type
Data Models:
    Foo: StructModel
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


@pytest.mark.parametrize(
    "cc, expected", [("sm_70", False), ("sm_86", True), ("sm_90", True)]
)
def test_cli_yml_inputs_full_spec_with_cc(tmpdir, kernel, cc, expected):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude: {{}}
Types:
    Foo: Type
Data Models:
    Foo: StructModel
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--compute-capability",
            cc,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    assert ("mul" in bindings) is expected


def test_yaml_deduce_missing_types(tmpdir, kernel):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude: {{}}
Types: {{}}
Data Models:
    Foo: StructModel
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_yaml_deduce_missing_datamodels(tmpdir, kernel):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude: {{}}
Types:
    Foo: Type
Data Models:
    {{}}
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_yaml_exclude_function(tmpdir):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude:
    Function:
        - add
Types:
    Foo: Type
Data Models:
    Foo: StructModel
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    assert "add" not in globals

    Foo = globals["Foo"]

    @cuda.jit
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.x

    arr = np.array([42])
    kernel[1, 1](arr)
    assert arr[0] == 0


def test_yaml_exclude_function_empty_list(tmpdir, kernel):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude:
    Function:
Types:
    Foo: Type
Data Models:
    Foo: StructModel
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_yaml_exclude_struct(tmpdir):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude:
    Struct:
        - Foo
Types: {{}}
Data Models: {{}}
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    assert "Foo" not in globals

    add = globals["add"]

    @cuda.jit
    def kernel(arr):
        arr[0] = add(1, 2)

    arr = np.array([42])
    kernel[1, 1](arr)
    assert arr[0] == 3


def test_yaml_exclude_struct_empty_list(tmpdir, kernel):
    clear_base_renderer_cache()
    clear_function_apis_registry()

    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")

    cfg = f"""Name: Test Data
Version: 0.0.1
Entry Point: {data}
File List:
    - {data}
Exclude:
    Struct:
Types: {{}}
Data Models: {{}}
"""

    cfg_file = subdir / "cfg.yaml"
    with open(cfg_file, "w") as f:
        f.write(cfg)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_file,
            "--output-dir",
            subdir,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output) as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)
