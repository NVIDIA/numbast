# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

from click.testing import CliRunner

import numpy as np
from numba import cuda
import pytest

from numbast.tools.static_binding_generator import static_binding_generator


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


def test_simple_cli(tmpdir):
    subdir = tmpdir.mkdir("sub")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
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

    Foo = globals["Foo"]
    add = globals["add"]
    c_ext_shim_source = globals["c_ext_shim_source"]

    @cuda.jit(link=[c_ext_shim_source])
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.x
        arr[1] = add(1, 2)

    arr = np.array([42, 0])
    kernel[1, 1](arr)
    assert arr[0] == 0
    assert arr[1] == 3


def test_simple_cli_retain(tmpdir):
    subdir = tmpdir.mkdir("sub2")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
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

    Foo = globals["Foo"]
    add = globals["add"]
    c_ext_shim_source = globals["c_ext_shim_source"]

    @cuda.jit(link=[c_ext_shim_source])
    def kernel(arr):
        foo = Foo()
        arr[0] = foo.x
        arr[1] = add(1, 2)

    arr = np.array([42, 0])
    kernel[1, 1](arr)
    assert arr[0] == 0
    assert arr[1] == 3


def test_simple_cli_no_retain(tmpdir):
    subdir = tmpdir.mkdir("sub3")
    data = os.path.join(os.path.dirname(__file__), "data.cuh")
    runner = CliRunner()

    false_path = subdir / "false.cuh"
    result = runner.invoke(
        static_binding_generator,
        [data, "--output-dir", subdir, "--retain", false_path],
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
