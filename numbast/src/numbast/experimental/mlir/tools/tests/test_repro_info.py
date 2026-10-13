# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_repro_info(run_in_isolated_folder, arch_str):
    """Consider both the config and output folder are traversible in the same
    tree, this PR makes sure that reproducible info accurately reflect where
    the config file can be located as a relative path to the binding file.
    """

    res = run_in_isolated_folder(
        "cfg.yml.j2", "data.cuh", {"arch_str": arch_str}, ruff_format=True
    )

    result = res["result"]
    binding_path = res["binding_path"]

    assert result.exit_code == 0

    with open(binding_path) as f:
        bindings = f.readlines()

    expected_info = {
        "Ast_canopy version",
        "Numbast version",
        "Generation command",
        "Static binding generator parameters",
        "Config file path (relative to the path of the generated binding)",
        "Cudatoolkit version",
    }

    # Check that all expected info are present within the generated binding in
    # the form of line comments.
    for line in bindings:
        if not expected_info:
            break

        if line.startswith("#"):
            comment = line[1:].strip()
            if ":" in comment:
                keys = comment.split(":")
                for k in keys:
                    expected_info.discard(k)

    assert len(expected_info) == 0


def _stamped_params(binding: str) -> str:
    marker = "# Static binding generator parameters: "
    for line in binding.splitlines():
        if line.startswith(marker):
            return line[len(marker) :]
    raise AssertionError("no parameter stamp in the generated binding")


def test_the_parameter_stamp_omits_options_left_at_their_default(
    run_in_isolated_folder, arch_str
):
    """The stamp describes the invocation, not the CLI's option list.

    See the fuller note on the copy of this test in ``numbast.tools``.
    """
    res = run_in_isolated_folder(
        "cfg.yml.j2",
        "data.cuh",
        {"arch_str": arch_str},
        omit_optional_options=True,
    )
    params = _stamped_params(res["binding"])

    assert "cfg_path" in params, params
    assert "output_dir" in params, params
    assert "run_ruff_format" not in params, params
    assert "bypass_parse_error" not in params, params


def test_the_parameter_stamp_keeps_options_that_were_passed(
    run_in_isolated_folder, arch_str
):
    res = run_in_isolated_folder(
        "cfg.yml.j2", "data.cuh", {"arch_str": arch_str}
    )
    params = _stamped_params(res["binding"])

    assert "run_ruff_format" in params, params
    assert "bypass_parse_error" in params, params
