# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import os
import json
from collections import defaultdict
import sys
import subprocess
import importlib
import warnings
import re

import yaml

from numba import config
import numba.types
import numba.core.datamodel.models

from ast_canopy import parse_declarations_from_source
from ast_canopy.decl import Function, Struct
from ast_canopy.pylibastcanopy import Enum, Typedef

from numbast.static import reset_renderer
from numbast.static.renderer import (
    get_shim,
    get_rendered_imports,
    get_reproducible_info,
    get_all_exposed_symbols,
    registry_setup,
)
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
)
from numbast.static.enum import StaticEnumsRenderer
from numbast.static.typedef import render_aliases
from numbast.tools.yaml_tags import string_constructor

config.CUDA_USE_NVIDIA_BINDING = True

VERBOSE = True

# Register custom YAML constructor for !join tag
yaml.add_constructor("!numbast_join", string_constructor)


class Config:
    """Configuration File for Static Binding Generation.

    Attributes
    ----------
    entry_point : str
        Path to the input CUDA header file.
    gpu_arch: list[str]
        The list of GPU architectures to generate bindings for. Currently, only
        one architecture per run is supported. Must be under pattern
        `sm_<compute_capability>`. Required.
    retain_list : list[str]
        List of file names to keep parsing. The list of files from which the
        declarations are retained in the final generated binding output. Bindings
        that exist in other source, which may get transitively included in the
        declaration, are ignored in bindings output.
    types : dict[str, type]
        A dictionary that maps struct names to their Numba types.
    datamodels : dict[str, type]
        A dictionary that maps struct names to their Numba data models.
    exclude_functions : list[str]
        List of function names to exclude from the bindings.
    exclude_structs : list[str]
        List of struct names to exclude from the bindings.
    clang_includes_paths : list[str]
        List of additional include paths to use when parsing the header file.
    additional_imports : list[str]
        The list of additional imports to add to the binding file.
    shim_include_override : str | None
        Override the include line of the shim function to specified string.
        If not specified, default to `#include <path_to_entry_point>`.
    predefined_macros : list[str]
        List of macros defined prior to parsing the header and prefixing shim functions.
    output_name : str | None
        The name of the output binding file, default None. When set to None, use
        the same name as input file (renamed with .py extension).
    cooperative_launch_required_functions_regex : list[str]
        The list of regular expressions. When any function name matches any of these
        regex patterns, the function should cause the kernel to be launched with
        cooperative launch.
    api_prefix_removal : dict[str, list[str]]
        Dictionary mapping declaration types to lists of prefixes to remove from names.
        For example, {"Function": ["prefix_"]} would remove "prefix_" from function names.
        Acceptable keywords: ["Struct", "Function", "Enum"]. Value types are lists of prefix
        strings. Specifically, prefixes in enums are also applicable to enum values.
    module_callbacks : dict[str, str]
        Dictionary containing setup and teardown callbacks for the module.
        Expected keys: "setup", "teardown". Each value is a string callback function.
    skip_prefix : str | None
        Do not generate bindings for any functions that start with this prefix.
        Has no effect if left unspecified.
    separate_registry : bool
        If true, use a separate typing and target registry for the generated binding.
        By default, the new typing and target registries are added to the existing
        typing and target context. When set to true, user should add the registries
        to the typing and target context manually. Default to False.
    """

    entry_point: str
    gpu_arch: list[str]
    retain_list: list[str]
    types: dict[str, type]
    datamodels: dict[str, type]
    exclude_functions: list[str]
    exclude_structs: list[str]
    clang_includes_paths: list[str]
    additional_imports: list[str]
    shim_include_override: str | None
    predefined_macros: list[str]
    output_name: str | None
    cooperative_launch_required_functions_regex: list[str]
    api_prefix_removal: dict[str, list[str]]
    module_callbacks: dict[str, str]
    skip_prefix: str | None
    separate_registry: bool

    def __init__(self, config_dict: dict):
        """Initialize Config from a dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing configuration values.
        """
        self.entry_point = config_dict["Entry Point"]
        self.gpu_arch = config_dict["GPU Arch"]
        self.retain_list = config_dict["File List"]
        self.types = _str_value_to_numba_type(config_dict.get("Types", {}))
        self.datamodels = _str_value_to_numba_datamodel(
            config_dict.get("Data Models", {})
        )

        self.excludes = config_dict.get("Exclude", {})
        self.exclude_functions = self.excludes.get("Function", [])
        self.exclude_structs = self.excludes.get("Struct", [])

        self.clang_includes_paths = config_dict.get("Clang Include Paths", [])

        self.additional_imports = config_dict.get("Additional Import", [])

        self.shim_include_override = config_dict.get(
            "Shim Include Override", None
        )

        self.predefined_macros = config_dict.get("Predefined Macros", [])

        if self.exclude_functions is None:
            self.exclude_functions = []
        if self.exclude_structs is None:
            self.exclude_structs = []
        if self.clang_includes_paths is None:
            self.clang_includes_paths = []

        self.output_name = config_dict.get("Output Name", None)

        self.cooperative_launch_required_functions_regex = config_dict.get(
            "Cooperative Launch Required Functions Regex", []
        )

        self.api_prefix_removal = config_dict.get("API Prefix Removal", {})

        # Ensure prefix removal values are lists
        if self.api_prefix_removal:
            for key, value in self.api_prefix_removal.items():
                if not isinstance(value, list):
                    self.api_prefix_removal[key] = [value]

        self.module_callbacks = config_dict.get("Module Callbacks", {})
        self.skip_prefix = config_dict.get("Skip Prefix", None)

        self.separate_registry = config_dict.get("Use Separate Registry", False)

        # TODO: support multiple GPU architectures
        if len(self.gpu_arch) > 1:
            raise NotImplementedError(
                "Multiple GPU architectures are not supported yet."
            )

        self._verify_exists()
        self._verify_regex_patterns()

    @classmethod
    def from_yaml_path(cls, cfg_path: str) -> "Config":
        """Create a Config instance from a YAML file path.

        Parameters
        ----------
        cfg_path : str
            Path to the YAML configuration file.

        Returns
        -------
        Config
            A new Config instance.
        """
        with open(cfg_path) as f:
            config_dict = yaml.load(f, yaml.Loader)
        return cls(config_dict)

    @classmethod
    def from_params(
        cls,
        entry_point: str,
        gpu_arch: list[str],
        retain_list: list[str],
        types: dict[str, type],
        datamodels: dict[str, type],
        exclude_functions: list[str] | None = None,
        exclude_structs: list[str] | None = None,
        clang_includes_paths: list[str] | None = None,
        additional_imports: list[str] | None = None,
        shim_include_override: str | None = None,
        predefined_macros: list[str] | None = None,
        output_name: str | None = None,
        cooperative_launch_required_functions_regex: list[str] | None = None,
        api_prefix_removal: dict[str, list[str]] | None = None,
        module_callbacks: dict[str, str] | None = None,
        skip_prefix: str | None = None,
        separate_registry: bool = False,
    ) -> "Config":
        """Create a Config instance from individual parameters instead of a config file."""
        if types is None:
            raise ValueError("Types must be provided")
        if datamodels is None:
            raise ValueError("Data models must be provided")

        config_dict = {
            "Entry Point": entry_point,
            "GPU Arch": gpu_arch,
            "File List": retain_list,
            "Types": {},
            "Data Models": {},
            "Exclude": {
                "Function": exclude_functions or [],
                "Struct": exclude_structs or [],
            },
            "Clang Include Paths": clang_includes_paths or [],
            "Additional Import": additional_imports or [],
            "Shim Include Override": shim_include_override,
            "Predefined Macros": predefined_macros or [],
            "Output Name": output_name,
            "Cooperative Launch Required Functions Regex": cooperative_launch_required_functions_regex
            or [],
            "API Prefix Removal": api_prefix_removal or {},
            "Module Callbacks": module_callbacks or {},
            "Skip Prefix": skip_prefix,
            "Separate Registry": separate_registry,
        }

        # Convert types and datamodels back to string format for the dict
        if types:
            config_dict["Types"] = {k: v.__name__ for k, v in types.items()}
        if datamodels:
            config_dict["Data Models"] = {
                k: v.__name__ for k, v in datamodels.items()
            }

        instance = cls(config_dict)
        return instance

    def _verify_exists(self):
        if not os.path.exists(self.entry_point):
            raise ValueError(
                f"Input header file does not exist: {self.entry_point}"
            )
        for f in self.retain_list:
            if not os.path.exists(f):
                raise ValueError(f"File in retain list does not exist: {f}")
        for f in self.clang_includes_paths:
            if not os.path.exists(f):
                raise ValueError(f"File in include list does not exist: {f}")

    def _verify_regex_patterns(self):
        for pattern in self.cooperative_launch_required_functions_regex:
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {pattern}")


def _str_value_to_numba_type(d: dict[str, str]) -> dict[str, type]:
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
) -> dict[str, type]:
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
    """
    Group C++ typedef declarations by their underlying type name.

    Parameters:
        typedef_decls (list[Typedef]): Typedef declarations to process.

    Returns:
        dict[str, list[str]]: Mapping from an underlying type name to a list of alias names (typedef names).
    """
    aliases = defaultdict(list)
    for typedef in typedef_decls:
        aliases[typedef.underlying_name].append(typedef.name)

    return aliases


def _generate_structs(
    struct_decls,
    header_path,
    types,
    data_models,
    struct_prefix_removal,
    excludes,
):
    """
    Render struct declarations into the Python source for struct bindings.

    Parameters:
        struct_decls (list): List of struct declaration objects to render.
        header_path (str): Path to the original header file associated with the declarations.
        types (dict): Mapping from struct name to corresponding numba type object (or None).
        data_models (dict): Mapping from struct name to corresponding numba datamodel object (or None).
        struct_prefix_removal (list): List of name prefixes to remove from struct identifiers when rendering.
        excludes (list): List of struct names to exclude from rendering.

    Returns:
        str: Rendered source code for the struct bindings.
    """
    specs = {}
    for struct_decl in struct_decls:
        struct_name = struct_decl.name
        this_type = types.get(struct_name, None)
        this_data_model = data_models.get(struct_name, None)
        specs[struct_name] = (this_type, this_data_model, header_path)

    SSR = StaticStructsRenderer(
        struct_decls,
        specs,
        struct_prefix_removal=struct_prefix_removal,
        excludes=excludes,
    )

    return SSR.render_as_str(with_imports=False, with_shim_stream=False)


def _generate_functions(
    func_decls: list[Function],
    header_path: str,
    excludes: list[str],
    cooperative_launch_functions: list[str],
    function_prefix_removal: list[str],
    skip_prefix: str | None,
) -> str:
    """
    Render Python bindings for the provided function declarations.

    Parameters:
        func_decls (list[Function]): Parsed function declarations to render.
        header_path (str): Path to the original header file used for the shim stream.
        excludes (list[str]): Function names to exclude from rendering.
        cooperative_launch_functions (list[str]): Regex patterns or exact names identifying functions that require cooperative launch handling.
        function_prefix_removal (list[str]): List of prefixes to strip from function names when generating bindings.
        skip_prefix (str | None): If provided, skip generating bindings for functions whose names start with this prefix.

    Returns:
        binding_source (str): Generated source code for the functions section (imports and shim stream are omitted).
    """

    SFR = StaticFunctionsRenderer(
        func_decls,
        header_path,
        excludes=excludes,
        cooperative_launch_required=cooperative_launch_functions,
        function_prefix_removal=function_prefix_removal,
        skip_prefix=skip_prefix,
    )

    return SFR.render_as_str(with_imports=False, with_shim_stream=False)


def _generate_enums(
    enum_decls: list[Enum], enum_prefix_removal: list[str] = []
):
    """
    Render enum declarations into binding source code.

    Parameters:
        enum_decls (list[Enum]): Parsed enum declarations to render.
        enum_prefix_removal (list[str]): Prefixes to remove from enum names before rendering.

    Returns:
        str: The rendered enum bindings as a source string.
    """
    SER = StaticEnumsRenderer(enum_decls, enum_prefix_removal)
    return SER.render_as_str(with_imports=False, with_shim_stream=False)


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
    config: Config,
    output_dir: str,
    log_generates: bool = False,
    cfg_file_path: str | None = None,
    sbg_params: dict[str, str] = {},
    bypass_parse_error: bool = False,
) -> str:
    """
    Generate static Python bindings for a CUDA C++ header using the provided configuration.

    Parameters:
        config (Config): Configuration containing entry point, parsing and rendering options.
        output_dir (str): Directory where the generated binding file will be written.
        log_generates (bool): If True, print counts and lists of declarations that will be generated.
        cfg_file_path (str | None): Path to the config file used to produce these bindings (used for reproducible metadata); may be None.
        sbg_params (dict[str, str]): Extra parameters to include in the generator metadata.
        bypass_parse_error (bool): If True, continue generation when source parsing reports recoverable errors.

    Returns:
        str: Absolute path to the generated binding file.
    """
    try:
        basename = os.path.basename(config.entry_point)
        basename = basename.split(".")[0]
    except Exception:
        click.echo(f"Unable to extract base name from {config.entry_point}.")
        raise

    entry_point = os.path.abspath(config.entry_point)
    retain_list = [os.path.abspath(path) for path in config.retain_list]

    if len(config.gpu_arch) == 0:
        raise ValueError("At least one GPU architecture must be provided.")
    elif len(config.gpu_arch) > 1:
        raise NotImplementedError(
            "Multiple GPU architectures are not supported yet."
        )

    compute_capability = config.gpu_arch[0]

    # TODO: we don't have tests on different compute capabilities for the static binding generator yet.
    # This will be added in future PRs.
    decls = parse_declarations_from_source(
        entry_point,
        retain_list,
        compute_capability=compute_capability,
        additional_includes=config.clang_includes_paths,
        defines=config.predefined_macros,
        verbose=VERBOSE,
        bypass_parse_error=bypass_parse_error,
    )
    structs = decls.structs
    functions = decls.functions
    enums = decls.enums
    typedefs = [
        td
        for td in decls.typedefs
        if td.underlying_name not in config.exclude_structs
    ]

    if log_generates:
        log_files_to_generate(functions, structs, enums, typedefs)

    aliases = _typedef_to_aliases(typedefs)
    rendered_aliases = render_aliases(aliases)

    enum_bindings = _generate_enums(
        enums, config.api_prefix_removal.get("Enum", [])
    )
    struct_bindings = _generate_structs(
        structs,
        entry_point,
        config.types,
        config.datamodels,
        config.api_prefix_removal.get("Struct", []),
        config.exclude_structs,
    )

    function_bindings = _generate_functions(
        functions,
        entry_point,
        config.exclude_functions,
        config.cooperative_launch_required_functions_regex,
        config.api_prefix_removal.get("Function", []),
        config.skip_prefix,
    )

    registry_setup_str = registry_setup(config.separate_registry)

    if config.shim_include_override is not None:
        shim_include = f"'#include <' + {config.shim_include_override} + '>'"
    else:
        shim_include = f'"#include <{entry_point}>"'
    shim_stream_str = get_shim(
        shim_include=shim_include,
        predefined_macros=config.predefined_macros,
        module_callbacks=config.module_callbacks,
    )
    imports_str = get_rendered_imports(
        additional_imports=config.additional_imports
    )

    # Example: Save the processed output to the output directory
    if config.output_name is None:
        output_file = os.path.join(output_dir, f"{basename}.py")
    else:
        output_file = os.path.join(output_dir, config.output_name)

    # Full command line that generate the binding:
    cmd = " ".join(sys.argv)

    # Compute the relative path from generated binding to the config file:
    if cfg_file_path is not None:
        config_rel_path = os.path.relpath(cfg_file_path, output_file)
    else:
        config_rel_path = "<not available>"

    exposed_symbols = get_all_exposed_symbols()

    assembled = f"""
# Automatically generated by Numbast Static Binding Generator
# Generator Information:
{get_reproducible_info(config_rel_path, cmd, sbg_params)}

# Imports:
{imports_str}
{registry_setup_str}
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

# Symbols:
{exposed_symbols}
"""

    with open(output_file, "w") as file:
        file.write(assembled)
        click.echo(
            f"Bindings for {config.entry_point} generated in {output_file}"
        )

    return output_file


def ruff_format_binding_file(binding_file_path: str):
    if not os.path.exists(binding_file_path):
        return

    subprocess.run(
        ["ruff", "check", "--select", "I", "--fix", binding_file_path],
        check=True,
    )

    print("Formatted.")


@click.command()
@click.pass_context
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
    "-fmt",
    "--run-ruff-format",
    type=bool,
    default=True,
)
@click.option(
    "-noraise",
    "--bypass-parse-error",
    type=bool,
    default=False,
)
def static_binding_generator(
    ctx,
    cfg_path,
    output_dir,
    run_ruff_format,
    bypass_parse_error,
):
    """
    A CLI tool to generate CUDA static bindings for CUDA C++ headers.

    CFG_PATH: Path to the configuration file in YAML format.
    OUTPUT_DIR: Path to the output directory where the processed files will be saved.
    RUN_RUFF_FORMAT: Run ruff format on the generated binding file.
    BYPASS_PARSE_ERROR: Bypass parse error and continue generating bindings.
    """
    reset_renderer()

    cfg = Config.from_yaml_path(cfg_path)
    output_file = _static_binding_generator(
        cfg,
        output_dir,
        log_generates=True,
        cfg_file_path=cfg_path,
        sbg_params=ctx.params,
        bypass_parse_error=bypass_parse_error,
    )

    if run_ruff_format:
        spec = importlib.util.find_spec("ruff")
        if spec is None:
            warnings.warn("Ruff is not on the system. Formatting skipped.")
        else:
            ruff_format_binding_file(output_file)
