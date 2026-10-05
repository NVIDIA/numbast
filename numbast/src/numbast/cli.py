from importlib import import_module

import click
import yaml

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

    use_mlir = bool(
        config.get("MLIR Backend", config.get("mlir_backend", False))
    )
    module_name = (
        "numbast.experimental.mlir.tools.static_binding_generator"
        if use_mlir
        else "numbast.tools.static_binding_generator"
    )
    generator = import_module(module_name).static_binding_generator
    return ctx.forward(generator)
