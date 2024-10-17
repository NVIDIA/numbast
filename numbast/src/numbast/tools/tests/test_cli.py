# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from click.testing import CliRunner

import numpy as np
from numba import cuda
import pytest

from numbast.tools.static_binding_generator import static_binding_generator


@pytest.fixture
def kernel():
    def _lazy_kernel(globals, cc=None):
        if cc is not None:
            compute_capability = int(cc[3:])
        else:
            cc = cuda.get_current_device().compute_capability
            compute_capability = cc[0] * 10 + cc[1]

        Foo = globals["Foo"]
        add = globals["add"]
        if compute_capability >= 86:
            mul = globals["mul"]

        c_ext_shim_source = globals["c_ext_shim_source"]

        if compute_capability >= 86:

            @cuda.jit(link=[c_ext_shim_source])
            def kernel(arr):
                foo = Foo()
                arr[0] = foo.x
                arr[1] = add(1, 2) + mul(3, 4)

        else:

            @cuda.jit(link=[c_ext_shim_source])
            def kernel(arr):
                foo = Foo()
                arr[0] = foo.x
                arr[1] = add(1, 2)

        arr = np.array([42, 0])
        kernel[1, 1](arr)
        assert arr[0] == 0

        if compute_capability >= 86:
            assert arr[1] == 15
        else:
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
    runner = CliRunner()
    with pytest.raises(Exception):
        result = runner.invoke(static_binding_generator, inputs)
        assert result.exit_code == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--input-header", "data.cuh"],
        ["--types", '{"Foo":"Type"}'],
        ["--datamodels", '{"Foo": "StructModel"}'],
    ],
)
def test_cli_yml_invalid_inputs(tmpdir, args):
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


def test_simple_cli(tmpdir, kernel):
    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--input-header",
            data,
            "--output-dir",
            subdir,
            "--types",
            '{"Foo":"Type"}',
            "--datamodels",
            '{"Foo": "StructModel"}',
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_simple_cli_retain(tmpdir, kernel):
    subdir = tmpdir.mkdir("sub2")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--input-header",
            data,
            "--output-dir",
            subdir,
            "--retain",
            data,
            "--types",
            '{"Foo":"Type"}',
            "--datamodels",
            '{"Foo": "StructModel"}',
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_simple_cli_no_retain(tmpdir):
    subdir = tmpdir.mkdir("sub3")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()

    false_path = subdir / "false.cuh"
    result = runner.invoke(
        static_binding_generator,
        ["--input-header", data, "--output-dir", subdir, "--retain", false_path],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    # Both are not in the retain list
    assert "Foo" not in globals
    assert "add" not in globals


@pytest.mark.parametrize(
    "cc, expected", [("sm_70", False), ("sm_86", True), ("sm_90", True)]
)
def test_simple_cli_compute_capability(tmpdir, cc, expected):
    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--input-header",
            data,
            "--output-dir",
            subdir,
            "--types",
            '{"Foo":"Type"}',
            "--datamodels",
            '{"Foo": "StructModel"}',
            "--compute-capability",
            cc,
        ],
    )

    assert result.exit_code == 0, f"CMD ERROR: {result.stdout}"

    output = subdir / "data.py"
    assert os.path.exists(output)

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    print(globals.keys())
    assert ("mul" in globals) is expected


@pytest.mark.skip("TODO: A C++ error is thrown.")
def test_simple_cli_empty_retain(tmpdir):
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

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    # Both are not in the retain list
    assert "Foo" not in globals
    assert "add" not in globals


@pytest.mark.parametrize("cc", ["sm_70", "sm_86", "sm_90"])
def test_cli_yml_inputs_full_spec(tmpdir, kernel, cc):
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

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals, cc)


def test_yaml_deduce_missing_types(tmpdir, kernel):
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

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_yaml_deduce_missing_datamodels(tmpdir, kernel):
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

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    kernel(globals)


def test_yaml_exclude_function(tmpdir):
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

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    assert "add" not in globals

    Foo = globals["Foo"]
    c_ext_shim_source = globals["c_ext_shim_source"]

    @cuda.jit(link=[c_ext_shim_source])
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.x

    arr = np.array([42])
    kernel[1, 1](arr)
    assert arr[0] == 0


def test_yaml_exclude_struct(tmpdir):
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

    with open(output, "r") as f:
        bindings = f.read()

    globals = {}
    exec(bindings, globals)

    assert "Foo" not in globals

    add = globals["add"]
    c_ext_shim_source = globals["c_ext_shim_source"]

    @cuda.jit(link=[c_ext_shim_source])
    def kernel(arr):
        arr[0] = add(1, 2)

    arr = np.array([42])
    kernel[1, 1](arr)
    assert arr[0] == 3
