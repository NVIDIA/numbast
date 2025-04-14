# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import os
import json
from collections import defaultdict
import sys
import subprocess

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
    get_shim_stream_obj,
    get_rendered_imports,
    get_reproducible_info,
)
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
)
from numbast.static.enum import StaticEnumsRenderer
from numbast.static.typedef import render_aliases

config.CUDA_USE_NVIDIA_BINDING = True

VERBOSE = False

CUDA_INCLUDE_PATH = config.CUDA_INCLUDE_PATH
MACHINE_COMPUTE_CAPABILITY = cuda.get_current_device().compute_capability


class YamlConfig:
    input_header: str
    retain_list: list[str]
    types: dict[str, numba.types.Type]
    datamodels: dict[str, numba.core.datamodel.models.DataModel]
    exclude_functions: list[str]
    exclude_structs: list[str]
    clang_includes_paths: list[str]

    def __init__(self, cfg_path):
        with open(cfg_path) as f:
            config = yaml.load(f, yaml.Loader)
            self.input_header = config["Entry Point"]
            self.retain_list = config["File List"]
            self.types = _str_value_to_numba_type(config.get("Types", {}))
            self.datamodels = _str_value_to_numba_datamodel(
                config.get("Data Models", {})
            )

            self.excludes = config.get("Exclude", {})
            self.exclude_functions = self.excludes.get("Function", [])
            self.exclude_structs = self.excludes.get("Struct", [])

            self.clang_includes_paths = config.get("Clang Include Paths", [])

            # FIXME: We are pretending that the list of macro-expanded functions is the same
            # as the list of declarations with anonymous filenames. This is not necessarily
            # true.
            self.macro_expanded_function_prefixes = config.get(
                "Macro-expanded Function Prefixes", []
            )

            if self.exclude_functions is None:
                self.exclude_functions = []
            if self.exclude_structs is None:
                self.exclude_structs = []
            if self.clang_includes_paths is None:
                self.clang_includes_paths = []

        self._verify_exists()

    def _verify_exists(self):
        if not os.path.exists(self.input_header):
            raise ValueError(
                f"Input header file does not exist: {self.input_header}"
            )
        for f in self.retain_list:
            if not os.path.exists(f):
                raise ValueError(f"File in retain list does not exist: {f}")
        for f in self.clang_includes_paths:
            if not os.path.exists(f):
                raise ValueError(f"File in include list does not exist: {f}")


def _str_value_to_numba_type(d: dict[str, str]) -> dict[str, numba.types.Type]:
    """Converts string typed value to numba `types` objects"""
    return {k: getattr(numba.types, v) for k, v in d.items()}


class NumbaTypeDictType(click.ParamType):
    """`Click` input type for dictionary mapping struct to Numba type."""

    name = "numba_type_dict"

    def convert(self, value, param, ctx):
        try:
            d = json.loads(value)
        except Exception:
            self.fail(
                f"{self.name} parameter must be valid JSON string. Got {value}"
            )

        try:
            d = _str_value_to_numba_type(d)
        except Exception:
            self.fail(
                f"Unable to convert input type dictionary string into dict of numba types. Got {d}."
            )

        return d


numba_type_dict = NumbaTypeDictType()


def _str_value_to_numba_datamodel(
    d: dict[str, str],
) -> dict[str, numba.core.datamodel.models.DataModel]:
    """Converts string typed value to numba `datamodel` objects"""
    return {k: getattr(numba.core.datamodel.models, v) for k, v in d.items()}


class NumbaDataModelDictType(click.ParamType):
    """`Click` input type for dictionary mapping struct to Numba data model."""

    name = "numba_datamodel_type"

    def convert(self, value, param, ctx):
        try:
            d = json.loads(value)
        except Exception:
            self.fail(
                f"{self.name} parameter must be valid JSON string. Got {value}"
            )

        try:
            d = _str_value_to_numba_datamodel(d)
        except Exception:
            self.fail(
                f"Unable to convert input data model dictionary string into dict of numba data models. Got {d}."
            )

        return d


numba_datamodel_dict = NumbaDataModelDictType()


def _typedef_to_aliases(typedef_decls: list[Typedef]) -> dict[str, list[str]]:
    """Convert C++ typedef declarations into aliases.

    `typedef` declarations contains a 1-1 mapping from "name" to "underlying name".
    There can be multiple typedefs of the same underlying name.

    This function aggregates them so that each "underlying name" maps to all names,
    aka, its aliases.

    Parameter
    ---------
    typedef_decls: list[Typedef]
        A list of C++ typedef declarations

    Return
    ------
    aliases: dict[str, list[str]]
        Dictionary mapping underlying names to a list of aliases.
    """
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
            f"  - {typedef.name}: {typedef.underlying_name}"
            for typedef in typedefs
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
    clang_include_paths: list[str],
    anon_filename_decl_prefix_allowlist: list[str],
    log_generates: bool = False,
    cfg_file_path: str | None = None,
    sbg_params: dict[str, str] = {},
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
    - clang_include_paths (list[str]): List of additional include paths to use when parsing the header file.
    - anon_filename_decl_prefix_allowlist (list[str]): List of prefixes to allow for anonymous filename declarations.
    - log_generates (bool, optional): Flag to log the list of bindings to generate. Defaults to False.
    - cfg_file_path (str, optional): Path to the configuration file. Defaults to None.
    - sbg_params (dict, optional): A dictionary of parameters for the static binding generator. Defaults to empty dict.

    Returns:
    None
    """
    try:
        basename = os.path.basename(entry_point)
        basename = basename.split(".")[0]
    except Exception:
        click.echo(f"Unable to extract base name from {entry_point}.")
        return

    entry_point = os.path.abspath(entry_point)
    retain_list = [os.path.abspath(path) for path in retain_list]

    # TODO: we don't have tests on different compute capabilities for the static binding generator yet.
    # This will be added in future PRs.
    decls = parse_declarations_from_source(
        entry_point,
        retain_list,
        compute_capability=compute_capability,
        cudatoolkit_include_dir=CUDA_INCLUDE_PATH,
        additional_includes=clang_include_paths,
        anon_filename_decl_prefix_allowlist=anon_filename_decl_prefix_allowlist,
        verbose=VERBOSE,
    )
    structs = decls.structs
    functions = decls.functions
    enums = decls.enums
    typedefs = decls.typedefs

    if log_generates:
        log_files_to_generate(functions, structs, enums, typedefs)

    aliases = _typedef_to_aliases(typedefs)
    rendered_aliases = render_aliases(aliases)

    enum_bindings = _generate_enums(enums)
    struct_bindings = _generate_structs(
        structs, entry_point, types, datamodels, exclude_structs
    )

    function_bindings = _generate_functions(
        functions, entry_point, exclude_functions
    )

    prefix_str = get_prefix()
    shim_stream_str = get_shim_stream_obj(header=entry_point)
    imports_str = get_rendered_imports()

    # Example: Save the processed output to the output directory
    output_file = os.path.join(output_dir, f"{basename}.py")

    # Full command line that generate the binding:
    cmd = " ".join(sys.argv)

    # Compute the relative path from generated binding to the config file:
    if cfg_file_path is not None:
        config_rel_path = os.path.relpath(cfg_file_path, output_file)
    else:
        config_rel_path = "<not available>"

    assembled = f"""
# Automatically generated by Numbast Static Binding Generator
# Generator Information:
{get_reproducible_info(config_rel_path, cmd, sbg_params)}

# Imports:
{imports_str}
# Prefixes:
{prefix_str}
# Shim Stream:
{shim_stream_str}
# Enums:
{enum_bindings}
# Structs:
{struct_bindings}
# Functions:
{function_bindings}
# Aliases:
{rendered_aliases}
"""

    with open(output_file, "w") as file:
        file.write(assembled)
        click.echo(f"Bindings for {entry_point} generated in {output_file}")

    return output_file


def ruff_format_binding_file(binding_file_path: str):
    if not os.path.exists(binding_file_path):
        return

    subprocess.run(["ruff", "check", "--fix", binding_file_path], check=True)

    print("Formatted.")


@click.command()
@click.pass_context
@click.option(
    "--input-header",
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option("--retain")
@click.option("--types", type=numba_type_dict)
@click.option("--datamodels", type=numba_datamodel_dict)
@click.option(
    "--cfg-path", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--output-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        writable=True,
    ),
    required=True,
)
@click.option(
    "--compute-capability",
    type=str,
    default=None,
)
@click.option(
    "--run-ruff-format",
    type=bool,
    default=True,
)
def static_binding_generator(
    ctx,
    input_header,
    retain,
    cfg_path,
    output_dir,
    types,
    datamodels,
    compute_capability,
    run_ruff_format,
):
    """
    A CLI tool to generate CUDA static bindings for CUDA C++ headers.

    INPUT_HEADER: Path to the input CUDA header file.
    CFG_PATH: Path to the configuration file in YAML format. If specified, only COMPUTE_CAPABILITY and OUTPUT_DIR is allowed as parameter.
    RETAIN: Comma separated list of file names to keep parsing, default to INPUT_HEADER.
    OUTPUT_DIR: Path to the output directory where the processed files will be saved.
    TYPES: A dictionary in JSON string that maps name of the struct to their Numba type.
    DATAMODELS: A dictionary in JSON string that maps name of the struct to their Numba datamodel.
    COMPUTE_CAPABILITY: Compute capability of the CUDA device, default to the current machine's compute capability.
    RUN_RUFF_FORMAT: Run ruff format on the generated binding file.
    """
    reset_renderer()

    if compute_capability is None:
        compute_capability = (
            f"sm_{MACHINE_COMPUTE_CAPABILITY[0]}{MACHINE_COMPUTE_CAPABILITY[1]}"
        )

    if not compute_capability.startswith("sm_"):
        raise ValueError("Compute capability must start with `sm_`")

    if cfg_path:
        if any(
            x is not None for x in [input_header, retain, types, datamodels]
        ):
            raise ValueError(
                "When CFG_PATH specified, none of INPUT_HEADER, RETAIN, TYPES and DATAMODELS should be specified."
            )

        cfg = YamlConfig(cfg_path)
        output_file = _static_binding_generator(
            cfg.input_header,
            cfg.retain_list,
            output_dir,
            cfg.types,
            cfg.datamodels,
            compute_capability,
            cfg.exclude_functions,
            cfg.exclude_structs,
            cfg.clang_includes_paths,
            cfg.macro_expanded_function_prefixes,
            cfg_file_path=cfg_path,
            sbg_params=ctx.params,
        )

        if run_ruff_format:
            ruff_format_binding_file(output_file)

        return

    if retain is None:
        retain_list = [input_header]
    else:
        retain_list = retain.split(",")

    if len(retain_list) == 0:
        raise ValueError("At least one file name to retain must be provided.")

    output_file = _static_binding_generator(
        input_header,
        retain_list,
        output_dir,
        types,
        datamodels,
        compute_capability,
        [],  # TODO: parse excludes from input
        [],  # TODO: parse excludes from input
    )

    if run_ruff_format:
        ruff_format_binding_file(output_file)
