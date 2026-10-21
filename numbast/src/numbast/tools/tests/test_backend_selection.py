# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which backend a configuration selects, from the CLI down to ``Config``."""

import pytest
import yaml
from click.testing import CliRunner

from numbast.backends.registry import (
    BACKEND_CONFIG_KEY,
    NUMBA_CUDA,
    NUMBA_CUDA_MLIR,
    BackendNotFoundError,
)
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
        raise AssertionError(f"routed to MLIR despite {NUMBA_CUDA}")

    monkeypatch.setattr(
        sbg, "_run_mlir_static_binding_generator", must_not_be_called
    )

    cfg_path = _cfg(tmp_path, **{BACKEND_CONFIG_KEY: NUMBA_CUDA})

    result = CliRunner().invoke(
        sbg.static_binding_generator,
        [
            "--cfg-path",
            str(cfg_path),
            "--output-dir",
            str(tmp_path),
            "-fmt",
            "false",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "input.py").exists()


def test_an_unknown_backend_fails_before_any_work(tmp_path):
    """Loudly, and without having written a partial tree to the output dir."""
    cfg_path = _cfg(tmp_path, **{BACKEND_CONFIG_KEY: "nonesuch"})

    with pytest.raises(BackendNotFoundError):
        CliRunner().invoke(
            sbg.static_binding_generator,
            [
                "--cfg-path",
                str(cfg_path),
                "--output-dir",
                str(tmp_path),
            ],
            catch_exceptions=False,
        )

    assert not (tmp_path / "input.py").exists()


def test_the_numba_driver_refuses_a_backend_it_cannot_serve(tmp_path):
    """Reachable only by invoking this driver directly, bypassing dispatch.

    Without the check, a registered backend nobody wired up here would fall
    through and emit Numba bindings under the file name the real backend would
    have written, which looks like success.
    """
    cfg_path = _cfg(tmp_path, **{BACKEND_CONFIG_KEY: "cuda-oxide"})

    result = CliRunner().invoke(
        sbg.static_binding_generator,
        ["--cfg-path", str(cfg_path), "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "cuda-oxide" in result.output
    assert not (tmp_path / "input.py").exists()


def test_config_exposes_the_resolved_backend(tmp_path):
    cfg_path = _cfg(tmp_path, **{BACKEND_CONFIG_KEY: NUMBA_CUDA_MLIR})
    config = sbg.Config.from_yaml_path(str(cfg_path))

    assert config.backend == NUMBA_CUDA_MLIR
    assert config.mlir_backend is True


def test_the_list_spelling_reaches_the_config_object(tmp_path):
    """``Config`` resolves the key itself, so it must read a list too."""
    cfg_path = _cfg(tmp_path, **{BACKEND_CONFIG_KEY: [NUMBA_CUDA_MLIR]})
    config = sbg.Config.from_yaml_path(str(cfg_path))

    assert config.backend == NUMBA_CUDA_MLIR
    assert config.mlir_backend is True
