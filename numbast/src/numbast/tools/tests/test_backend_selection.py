# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selection at the CLI, and what ends up in the provenance stamp."""

import pytest
import yaml
from click.testing import CliRunner

from numbast.backends.registry import BackendNotFoundError
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


def test_the_command_line_overrides_the_config(tmp_path, monkeypatch):
    """The point of the flag: try another backend without editing the recipe.

    Overriding *towards* the Numba backend rather than away from it, so the
    test exercises the whole CLI without needing a ``numba_cuda_mlir``
    installation.

    The MLIR entry point is replaced with one that fails rather than merely
    asserting that a binding appeared: both backends write the same file name,
    so a test that only checked for output would pass even if the override were
    dropped entirely.
    """

    def must_not_be_called(*args, **kwargs):
        raise AssertionError(
            "routed to the MLIR backend despite --backend numba"
        )

    monkeypatch.setattr(
        sbg, "_run_mlir_static_binding_generator", must_not_be_called
    )

    cfg_path = _cfg(tmp_path, Backend="numba-mlir")

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
            "--backend",
            "numba",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "input.py").exists()


def test_an_override_reaches_the_config_object_too(tmp_path):
    """Both dispatch points must agree within one run.

    The CLI decides which ``Config`` class to build by reading the YAML, but
    ``_static_binding_generator`` re-dispatches on ``config.mlir_backend``. If
    the override reached only the first, a run could pick the Numba config and
    then hand it to the MLIR generator.
    """
    cfg_path = _cfg(tmp_path, Backend="numba-mlir")
    config = sbg.Config.from_yaml_path(str(cfg_path), "numba")

    assert config.backend == "numba"
    assert config.mlir_backend is False


def test_an_unknown_backend_fails_before_any_work(tmp_path):
    """Loudly, and without having written a partial tree to the output dir."""
    cfg_path = _cfg(tmp_path)

    runner = CliRunner(catch_exceptions=False)
    with pytest.raises(BackendNotFoundError):
        runner.invoke(
            sbg.static_binding_generator,
            [
                "--cfg-path",
                str(cfg_path),
                "--output-dir",
                str(tmp_path),
                "--backend",
                "nonesuch",
            ],
        )

    assert not (tmp_path / "input.py").exists()


def test_config_exposes_the_resolved_backend(tmp_path):
    cfg_path = _cfg(tmp_path, Backend="numba-mlir")
    config = sbg.Config.from_yaml_path(str(cfg_path))

    assert config.backend == "numba-mlir"
    assert config.mlir_backend is True


def test_the_legacy_boolean_still_reaches_config(tmp_path):
    cfg_path = _cfg(tmp_path)
    cfg_path.write_text(
        cfg_path.read_text(encoding="utf-8") + "MLIR Backend: true\n",
        encoding="utf-8",
    )
    config = sbg.Config.from_yaml_path(str(cfg_path))

    assert config.backend == "numba-mlir"
    assert config.mlir_backend is True


