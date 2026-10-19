# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which backend a configuration selects, from the CLI down to ``Config``."""

import pytest
import yaml
from click.testing import CliRunner

from numbast.backends.registry import MLIR, NUMBA, BackendNotFoundError
from numbast.tools import static_binding_generator as sbg


def _cfg(tmp_path, **keys):
    header = tmp_path / "input.cuh"
    header.write_text("__device__ int f();\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "Entry Point": str(header),
                "GPU Arch": ["sm_80"],
                "File List": [str(header)],
                "Types": {},
                "Data Models": {},
                **keys,
            }
        ),
        encoding="utf-8",
    )
    return cfg_path


def test_the_config_selects_the_backend_end_to_end(tmp_path, monkeypatch):
    """Naming the Numba backend must not reach the MLIR entry point.

    The MLIR entry point is replaced with one that fails rather than merely
    asserting that a binding appeared: both backends write the same file name,
    so a test that only checked for output would pass even if dispatch were
    wrong.
    """

    def must_not_be_called(*args, **kwargs):
        raise AssertionError(f"routed to the MLIR backend despite {NUMBA}")

    monkeypatch.setattr(
        sbg, "_run_mlir_static_binding_generator", must_not_be_called
    )

    cfg_path = _cfg(tmp_path, Backend=NUMBA)

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(
        sbg.static_binding_generator,
        [
            "--cfg-path",
            str(cfg_path),
            "--output-dir",
            str(tmp_path),
            "-fmt",
            "false",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "input.py").exists()


def test_an_unknown_backend_fails_before_any_work(tmp_path):
    """Loudly, and without having written a partial tree to the output dir."""
    cfg_path = _cfg(tmp_path, Backend="nonesuch")

    runner = CliRunner(catch_exceptions=False)
    with pytest.raises(BackendNotFoundError):
        runner.invoke(
            sbg.static_binding_generator,
            [
                "--cfg-path",
                str(cfg_path),
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert not (tmp_path / "input.py").exists()


def test_config_exposes_the_resolved_backend(tmp_path):
    cfg_path = _cfg(tmp_path, Backend=MLIR)
    config = sbg.Config.from_yaml_path(str(cfg_path))

    assert config.backend == MLIR
    assert config.mlir_backend is True


def test_the_list_spelling_reaches_the_config_object(tmp_path):
    """``Config`` resolves the key itself, so it must read a list too."""
    cfg_path = _cfg(tmp_path, Backend=[MLIR])
    config = sbg.Config.from_yaml_path(str(cfg_path))

    assert config.backend == MLIR
    assert config.mlir_backend is True
