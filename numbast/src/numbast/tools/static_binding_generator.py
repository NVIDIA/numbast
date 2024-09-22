# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import os
import json
from collections import defaultdict

import yaml

import numba.types
import numba.core.datamodel.models

from ast_canopy import parse_declarations_from_source
from ast_canopy.decl import Function

from numbast.static.renderer import (
    get_prefix,
    get_rendered_shims,
    get_rendered_imports,
    clear_base_renderer_cache,
)
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
    clear_function_apis_registry,
)


def _str_value_to_numba_type(d: dict):
    """Converts string typed value to numba `types` objects"""
    d_copy = {}

    keys = [*d.keys()]
    for k in keys:
        d_copy[k] = getattr(numba.types, d[k])
    return d_copy


class NumbaTypeDictType(click.ParamType):
    """`Click` input type for dictionary mapping struct to Numba type."""

    name = "numba_type_dict"

    def convert(self, value, param, ctx):
        try:
            d = json.loads(value)
        except Exception:
            self.fail(f"{self.name} parameter must be valid JSON string. Got {value}")

        try:
            d = _str_value_to_numba_type(d)
        except Exception:
            self.fail(
                f"Unable to convert input type dictionary string into dict of numba types. Got {d}."
            )

        return d


numba_type_dict = NumbaTypeDictType()


def _str_value_to_numba_datamodel(d: dict):
    """Converts string typed value to numba `datamodel` objects"""
    d_copy = {}

    keys = [*d.keys()]
    for k in keys:
        d_copy[k] = getattr(numba.core.datamodel.models, d[k])
    return d_copy


class NumbaDataModelDictType(click.ParamType):
    """`Click` input type for dictionary mapping struct to Numba data model."""

    name = "numba_datamodel_type"

    def convert(self, value, param, ctx):
        try:
            d = json.loads(value)
        except Exception:
            self.fail(f"{self.name} parameter must be valid JSON string. Got {value}")

        try:
            d = _str_value_to_numba_datamodel(d)
        except Exception:
            self.fail(
                f"Unable to convert input data model dictionary string into dict of numba data models. Got {d}."
            )

        return d


numba_datamodel_dict = NumbaDataModelDictType()


def _generate_structs(struct_decls, aliases, header_path, types, data_models):
    """Convert CLI inputs into structure that fits `StaticStructsRenderer` and create struct bindings."""
    specs = {}
    for struct_decl in struct_decls:
        struct_name = struct_decl.name
        this_type = types.get(struct_name, None)
        this_data_model = data_models.get(struct_name, None)
        specs[struct_name] = (this_type, this_data_model, header_path)

    SSR = StaticStructsRenderer(struct_decls, specs)

    return SSR.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )


def _generate_functions(func_decls: list[Function], header_path: str) -> str:
    """Convert CLI inputs into structure that fits `StaticStructsRenderer` and create struct bindings."""

    SFR = StaticFunctionsRenderer(func_decls, header_path)

    return SFR.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )


def _static_binding_generator(
    entry_point: str,
    retain_list: list[str],
    output_dir: str,
    types: dict[str, type],
    datamodels: dict[str, type],
    compute_capability: str,
):
    try:
        basename = os.path.basename(entry_point)
        basename = basename.split(".")[0]
    except Exception:
        click.echo(f"Unable to extract base name from {entry_point}.")
        return

    # TODO: we don't have tests on different compute capabilities for the static binding generator yet.
    # This will be added in future PRs.
    structs, functions, function_templates, class_templates, typedefs, enums = (
        parse_declarations_from_source(
            entry_point, retain_list, compute_capability=compute_capability
        )
    )

    aliases = defaultdict(list)
    for typedef in typedefs:
        aliases[typedef.underlying_name].append(typedef.name)

    struct_bindings = _generate_structs(
        structs, aliases, entry_point, types, datamodels
    )

    function_bindings = _generate_functions(functions, entry_point)

    prefix_str = get_prefix()
    imports_str = get_rendered_imports()
    shim_function_str = get_rendered_shims()

    # Example: Save the processed output to the output directory
    output_file = os.path.join(output_dir, f"{basename}.py")
    with open(output_file, "w") as file:
        file.write(prefix_str)
        file.write("\n")
        file.write(imports_str)
        file.write("\n")
        file.write(struct_bindings)
        file.write("\n")
        file.write(function_bindings)
        file.write("\n")
        file.write(shim_function_str)
        click.echo(f"Bindings for {entry_point} generated in {output_file}")


@click.command()
@click.pass_context
@click.option(
    "--input-header", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option("--retain")
@click.option("--types", type=numba_type_dict)
@click.option("--datamodels", type=numba_datamodel_dict)
@click.option("--cfg-path", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--output-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        writable=True,
    ),
)
def static_binding_generator(
    ctx,
    input_header,
    retain,
    cfg_path,
    output_dir,
    types,
    datamodels,
):
    """
    A CLI tool to generate CUDA static bindings for CUDA C++ headers.

    INPUT_HEADER: Path to the input CUDA header file.
    RETAIN: Comma separated list of file names to keep parsing, default to INPUT_HEADER.
    OUTPUT_DIR: Path to the output directory where the processed files will be saved.
    TYPES: A dictionary in JSON string that maps name of the struct to their Numba type.
    DATAMODELS: A dictionary in JSON string that maps name of the struct to their Numba datamodel.
    """
    # TODO: We should support input of types and data models from an external spec file for better UX.

    # To handle multiple runs of the CLI tools in the same python session (e.g. pytest)
    clear_base_renderer_cache()
    clear_function_apis_registry()

    if cfg_path:
        if any(x is not None for x in [input_header, retain, types, datamodels]):
            raise ValueError(
                "When CFG_PATH specified, none of INPUT_HEADER, RETAIN, TYPES and DATAMODELS should be specified."
            )

        with open(cfg_path, "r") as f:
            config = yaml.load(f, yaml.Loader)
            input_header = config["Entry Point"]
            retain_list = config["File List"]
            types = _str_value_to_numba_type(config["Types"])
            datamodels = _str_value_to_numba_datamodel(config["Data Models"])

            _static_binding_generator(
                input_header,
                retain_list,
                output_dir,
                types,
                datamodels,
                compute_capability="sm_70",  # TODO: Use compute capability from cli input
            )

            return

    if retain is None:
        retain_list = [input_header]
    else:
        retain_list = retain.split(",")

    if len(retain_list) == 0:
        raise ValueError("At least one file name to retain must be provided.")

    _static_binding_generator(
        input_header,
        retain_list,
        output_dir,
        types,
        datamodels,
        compute_capability="sm_70",  # TODO: Use compute capability from cli input
    )
