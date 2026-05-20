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
