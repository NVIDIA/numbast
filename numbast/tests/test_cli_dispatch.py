import json
import subprocess
import sys
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

from numbast import cli


@pytest.mark.parametrize(
    ("config", "backend"),
    [
        ("{}", "numbast.tools"),
        ("MLIR Backend: false", "numbast.tools"),
        ("MLIR Backend: true", "numbast.experimental.mlir.tools"),
        ("mlir_backend: true", "numbast.experimental.mlir.tools"),
        (
            'MLIR Backend: !numbast_join ["tr", "ue"]',
            "numbast.experimental.mlir.tools",
        ),
        ("MLIR Backend: false\nmlir_backend: true", "numbast.tools"),
    ],
)
def test_dispatch_forwards_options(tmp_path, monkeypatch, config, backend):
    config_path = tmp_path / "config.yml"
    config_path.write_text(config)
    calls = []

    @click.command()
    @click.pass_context
    def generator(ctx, **options):
        assert ctx.params == options
        click.echo(json.dumps(options))

    def import_backend(name):
        calls.append(name)
        return SimpleNamespace(static_binding_generator=generator)

    monkeypatch.setattr(cli, "import_module", import_backend)
    result = CliRunner().invoke(
        cli.static_binding_generator,
        [
            f"--cfg-path={config_path}",
            "--output-dir",
            str(tmp_path),
            "-fmt",
            "false",
            "-noraise",
            "true",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [f"{backend}.static_binding_generator"]
    assert json.loads(result.output) == {
        "cfg_path": str(config_path),
        "output_dir": str(tmp_path),
        "run_ruff_format": False,
        "bypass_parse_error": True,
    }


@pytest.mark.parametrize("config", ["", "[]", "[", "!!python/object:dict {}"])
def test_invalid_config_does_not_import_backend(tmp_path, monkeypatch, config):
    config_path = tmp_path / "config.yml"
    config_path.write_text(config)

    def unexpected_import(name):
        pytest.fail(f"Imported backend for invalid config: {name}")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    result = CliRunner().invoke(
        cli.static_binding_generator,
        ["--cfg-path", str(config_path), "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "Invalid value for --cfg-path" in result.output


def test_module_help_does_not_import_backends():
    script = """
import importlib.abc
import runpy
import sys

class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numba', 'numba_cuda_mlir', 'ast_canopy'}:
            raise AssertionError(f'Unexpected backend import: {fullname}')

sys.meta_path.insert(0, BlockBackends())
sys.argv = ['numbast', '--help']
runpy.run_module('numbast', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--cfg-path" in result.stdout
