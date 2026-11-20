from importlib import import_module

import click
import yaml

from numbast.backends.registry import (
    CUDA_OXIDE,
    BackendNotFoundError,
    BackendNotRunnableError,
    MultipleBackendsError,
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
def static_binding_generator(
    ctx,
    cfg_path,
    output_dir,
    run_ruff_format,
    bypass_parse_error,
):
    """Generate CUDA static bindings using the backend selected in the config."""
    try:
        config = load_binding_config(cfg_path)
    except (TypeError, yaml.YAMLError) as error:
        raise click.BadParameter(str(error), param_hint="--cfg-path") from error

    try:
        backend = resolve_backend_name(config)
        module_name = generator_module(backend)
    except (
        BackendNotFoundError,
        BackendNotRunnableError,
        MultipleBackendsError,
    ) as error:
        raise click.BadParameter(str(error), param_hint="--cfg-path") from error

    module = import_module(module_name)
    if backend == CUDA_OXIDE:
        return ctx.invoke(
            module.cuda_oxide_binding_generator,
            cfg_path=cfg_path,
            output_dir=output_dir,
        )
    generator = module.static_binding_generator
    return ctx.forward(generator)
