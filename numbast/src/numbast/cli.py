from importlib import import_module

import click
import yaml

from numbast.backends.registry import (
    BackendNotFoundError,
    BackendNotRunnableError,
    available_backends,
    generator_module,
    resolve_backend_name,
)
from numbast.tools.config.shared import load_binding_config


@click.command()
@click.pass_context
@click.option(
    "--cfg-path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    required=True,
)
@click.option("-fmt", "--run-ruff-format", type=bool, default=True)
@click.option("-noraise", "--bypass-parse-error", type=bool, default=False)
@click.option(
    "--backend",
    type=str,
    default=None,
    help=(
        'Emission backend, overriding the config\'s "Backend" key. One of: '
        + ", ".join(available_backends())
    ),
)
def static_binding_generator(
    ctx,
    cfg_path,
    output_dir,
    run_ruff_format,
    bypass_parse_error,
    backend,
):
    """Generate CUDA static bindings using the backend selected in the config."""
    try:
        config = load_binding_config(cfg_path)
    except (TypeError, yaml.YAMLError) as error:
        raise click.BadParameter(str(error), param_hint="--cfg-path") from error

    try:
        name = resolve_backend_name(config, backend)
        module_name = generator_module(name)
    except (BackendNotFoundError, BackendNotRunnableError) as error:
        # Blame whichever input actually named the backend, so the message
        # points at the file when the file chose and at the flag when it did.
        hint = "--backend" if backend else "--cfg-path"
        raise click.BadParameter(str(error), param_hint=hint) from error

    generator = import_module(module_name).static_binding_generator
    # Forward the resolved name rather than the override, so the generator
    # cannot re-resolve it against the config and reach a different answer
    # than the one that chose it.
    return ctx.forward(generator, backend=name)
