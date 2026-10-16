import json
import subprocess
import sys
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

from numbast import cli
from numbast.provenance import params_the_user_set


def _stamp_reporting_generator():
    """A stand-in generator that reports its provenance stamp and nothing else.

    It declares the dispatcher's options so ``ctx.forward`` has something to
    fill. The real generators need a CUDA toolchain and a parseable header to
    answer a question that is purely about click's bookkeeping.
    """

    @click.command()
    @click.pass_context
    @click.option("--cfg-path", type=str, required=True)
    @click.option("--output-dir", type=str, required=True)
    @click.option("-fmt", "--run-ruff-format", type=bool, default=True)
    @click.option("-noraise", "--bypass-parse-error", type=bool, default=False)
    def generator(ctx, **options):
        click.echo(json.dumps(sorted(params_the_user_set(ctx))))

    return generator


def _dispatch_to(monkeypatch, generator):
    monkeypatch.setattr(
        cli,
        "import_module",
        lambda name: SimpleNamespace(static_binding_generator=generator),
    )


@pytest.mark.parametrize(
    ("optional_argv", "also_stamped"),
    [
        ([], []),
        (["-fmt", "false"], ["run_ruff_format"]),
        (["-noraise", "true"], ["bypass_parse_error"]),
        (
            ["-fmt", "false", "-noraise", "true"],
            ["bypass_parse_error", "run_ruff_format"],
        ),
    ],
)
def test_defaults_stay_out_of_the_stamp_through_the_dispatcher(
    tmp_path, monkeypatch, optional_argv, also_stamped
):
    config_path = tmp_path / "config.yml"
    config_path.write_text("{}")
    _dispatch_to(monkeypatch, _stamp_reporting_generator())

    result = CliRunner().invoke(
        cli.static_binding_generator,
        [
            "--cfg-path",
            str(config_path),
            "--output-dir",
            str(tmp_path),
            *optional_argv,
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == sorted(
        ["cfg_path", "output_dir", *also_stamped]
    )


def test_both_entry_points_stamp_the_same_options(tmp_path, monkeypatch):
    """The dispatcher is the only installed entry point, so if it disagrees
    with a direct invocation, the shipped command is the one that is wrong.

    It disagreed: ``ctx.forward`` builds the callee's context without parsing
    and so records no parameter sources, which made every declared option look
    like something the user had passed.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text("{}")
    generator = _stamp_reporting_generator()
    argv = [
        "--cfg-path",
        str(config_path),
        "--output-dir",
        str(tmp_path),
        "-fmt",
        "false",
    ]

    direct = CliRunner().invoke(generator, argv)
    _dispatch_to(monkeypatch, generator)
    dispatched = CliRunner().invoke(cli.static_binding_generator, argv)

    assert direct.exit_code == 0, direct.output
    assert dispatched.exit_code == 0, dispatched.output
    assert json.loads(dispatched.output) == json.loads(direct.output)
    # Pinned, because two entry points broken the same way also agree.
    assert json.loads(direct.output) == [
        "cfg_path",
        "output_dir",
        "run_ruff_format",
    ]


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
        ("Backend: numba", "numbast.tools"),
        ("Backend: mlir", "numbast.experimental.mlir.tools"),
        (
            "Backend: mlir\nMLIR Backend: false",
            "numbast.experimental.mlir.tools",
        ),
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
        # The resolved name, not the flag. A generator that re-read the config
        # would otherwise be free to disagree with the dispatch that chose it.
        "backend": "mlir" if "mlir" in backend else "numba",
    }


def test_the_flag_overrides_the_config_at_the_real_entry_point(
    tmp_path, monkeypatch
):
    """``--backend`` has to be declared here or it is unreachable.

    This is the only installed command, so an override the dispatcher does not
    accept is an override nobody can use.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text("Backend: mlir")
    calls = []

    @click.command()
    @click.pass_context
    def generator(ctx, **options):
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
            "--backend",
            "numba",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["numbast.tools.static_binding_generator"]
    assert json.loads(result.output)["backend"] == "numba"


@pytest.mark.parametrize(
    ("args", "config", "hint"),
    [
        # A name the user typed is blamed on the flag, one the file chose on
        # the file, so the message points at the thing worth editing.
        (["--backend", "nonesuch"], "{}", "--backend"),
        ([], "Backend: nonesuch", "--cfg-path"),
        (["--backend", "cuda-oxide"], "{}", "--backend"),
        ([], "Backend: cuda-oxide", "--cfg-path"),
    ],
)
def test_an_unusable_backend_fails_before_importing_one(
    tmp_path, monkeypatch, args, config, hint
):
    """Including ``cuda-oxide``, which is registered but has no driver.

    Importing first would make a missing driver indistinguishable from a
    backend whose dependencies are not installed.
    """
    config_path = tmp_path / "config.yml"
    config_path.write_text(config)

    def unexpected_import(name):
        pytest.fail(f"Imported a backend for an unusable name: {name}")

    monkeypatch.setattr(cli, "import_module", unexpected_import)
    result = CliRunner().invoke(
        cli.static_binding_generator,
        ["--cfg-path", str(config_path), "--output-dir", str(tmp_path), *args],
    )

    assert result.exit_code == 2
    assert hint in result.output


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
