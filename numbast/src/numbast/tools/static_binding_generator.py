# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import os
import json
from collections import defaultdict

import yaml

from numba import config, cuda
import numba.types
import numba.core.datamodel.models

from ast_canopy import parse_declarations_from_source
from ast_canopy.decl import Function, Struct
from pylibastcanopy import Enum, Typedef

from numbast.static import reset_renderer
from numbast.static.renderer import (
    get_prefix,
    get_rendered_shims,
    get_rendered_imports,
)
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
)
from numbast.static.enum import StaticEnumsRenderer
from numbast.static.typedef import render_aliases

config.CUDA_USE_NVIDIA_BINDING = True

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability


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


def _typedef_to_aliases(typedef_decls: list[Typedef]) -> dict[str, list[str]]:
    aliases = defaultdict(list)
    for typedef in typedef_decls:
        aliases[typedef.underlying_name].append(typedef.name)

    return aliases


def _generate_structs(struct_decls, header_path, types, data_models, excludes):
    """Convert CLI inputs into structure that fits `StaticStructsRenderer` and create struct bindings."""
    specs = {}
    for struct_decl in struct_decls:
        struct_name = struct_decl.name
        this_type = types.get(struct_name, None)
        this_data_model = data_models.get(struct_name, None)
        specs[struct_name] = (this_type, this_data_model, header_path)

    SSR = StaticStructsRenderer(struct_decls, specs, excludes=excludes)

    return SSR.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )


def _generate_functions(
    func_decls: list[Function], header_path: str, excludes: list[str]
) -> str:
    """Convert CLI inputs into structure that fits `StaticStructsRenderer` and create struct bindings."""

    SFR = StaticFunctionsRenderer(func_decls, header_path, excludes=excludes)

    return SFR.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )


def _generate_enums(enum_decls: list[Enum]):
    """Create enum bindings."""
    SER = StaticEnumsRenderer(enum_decls)
    return SER.render_as_str(
        with_prefix=False, with_imports=False, with_shim_functions=False
    )


def log_files_to_generate(
    functions: list[Function],
    structs: list[Struct],
    enums: list[Enum],
    typedefs: list[Typedef],
):
    """Console log the list of bindings to generate."""

    click.echo("-" * 80)
    click.echo(
        f"Generating bindings for {len(functions)} functions, {len(structs)} structs, {len(typedefs)} typedefs, {len(enums)} enums."
    )

    click.echo("Enums: ")
    click.echo("\n".join(f"  - {enum.name}" for enum in enums))
    click.echo("TypeDefs: ")
    click.echo(
        "\n".join(
            f"  - {typedef.name}: {typedef.underlying_name}" for typedef in typedefs
        )
    )
    click.echo("Functions: ")
    click.echo("\n".join(f"  - {str(func)}" for func in functions))
    click.echo("\nStructs: ")
    click.echo("\n".join(f"  - {struct.name}" for struct in structs))


def _static_binding_generator(
    entry_point: str,
    retain_list: list[str],
    output_dir: str,
    types: dict[str, type],
    datamodels: dict[str, type],
    compute_capability: str,
    exclude_functions: list[str],
    exclude_structs: list[str],
    log_generates: bool = False,
):
    """
    A function to generate CUDA static bindings for CUDA C++ headers.

    Parameters:
    - entry_point (str): Path to the input CUDA header file.
    - retain_list (list[str]): List of file names to keep parsing.
    - output_dir (str): Path to the output directory where the processed files will be saved.
    - types (dict[str, type]): A dictionary that maps struct names to their Numba types.
    - datamodels (dict[str, type]): A dictionary that maps struct names to their Numba data models.
    - compute_capability (str): Compute capability of the CUDA device.
    - exclude_functions (list[str]): List of function names to exclude from the bindings.
    - exclude_structs (list[str]): List of struct names to exclude from the bindings.
    - log_generates (bool, optional): Flag to log the list of bindings to generate. Defaults to False.

    Returns:
    None
    """
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
            entry_point,
            retain_list,
            compute_capability=compute_capability,
            cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
        )
    )

    if log_generates:
        log_files_to_generate(functions, structs, enums, typedefs)

    aliases = _typedef_to_aliases(typedefs)
    rendered_aliases = render_aliases(aliases)

    enum_bindings = _generate_enums(enums)
    struct_bindings = _generate_structs(
        structs, entry_point, types, datamodels, exclude_structs
    )

    function_bindings = _generate_functions(functions, entry_point, exclude_functions)

    prefix_str = get_prefix()
    imports_str = get_rendered_imports()
    shim_function_str = get_rendered_shims()

    # Example: Save the processed output to the output directory
    output_file = os.path.join(output_dir, f"{basename}.py")

    assembled = f"""
# Automatically generated by Numbast Static Binding Generator

# Prefixes:
{prefix_str}
# Imports:
{imports_str}
# Enums:
{enum_bindings}
# Structs:
{struct_bindings}
# Functions:
{function_bindings}
# Aliases:
{rendered_aliases}
# Shim functions:
{shim_function_str}
"""

    with open(output_file, "w") as file:
        file.write(assembled)
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
    reset_renderer()

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

            excludes = config["Exclude"]
            exclude_functions = excludes.get("Function", [])
            exclude_structs = excludes.get("Struct", [])

            _static_binding_generator(
                input_header,
                retain_list,
                output_dir,
                types,
                datamodels,
                f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",  # TODO: Use compute capability from cli input
                exclude_functions,
                exclude_structs,
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
        f"sm_{COMPUTE_CAPABILITY[0]}{COMPUTE_CAPABILITY[1]}",  # TODO: Use compute capability from cli input
        [],  # TODO: parse excludes from input
        [],  # TODO: parse excludes from input
    )
