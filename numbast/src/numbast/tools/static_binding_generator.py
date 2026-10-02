# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import click
import os
import json
from collections import defaultdict
import subprocess
import importlib
import warnings

from numba import config

from ast_canopy import parse_declarations_from_source
from ast_canopy.decl import (
    ClassTemplate,
    Function,
    FunctionTemplate,
    Struct,
)
from ast_canopy.pylibastcanopy import Enum, Typedef

from numbast.static import reset_renderer
from numbast.static.renderer import (
    get_shim,
    get_rendered_imports,
    get_reproducible_info,
    get_all_exposed_symbols,
    registry_setup,
    get_callconv_utils,
)
from numbast.static.struct import StaticStructsRenderer
from numbast.static.function import (
    StaticFunctionsRenderer,
)
from numbast.static.function_template import StaticFunctionTemplatesRenderer
from numbast.static.class_template import StaticClassTemplatesRenderer
from numbast.static.enum import StaticEnumsRenderer
from numbast.static.typedef import render_aliases
from numbast.tools.config.shared import (
    config_uses_mlir_backend,
    load_binding_config,
)
from numbast.tools.config.numba import (
    NumbaConfig,
    _str_value_to_numba_datamodel,
    _str_value_to_numba_type,
)

config.CUDA_USE_NVIDIA_BINDING = True

VERBOSE = True
STATIC_BINDING_CONFIG_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "static_binding_generator.schema.yaml"
)


def _config_dict_uses_mlir_backend(config_dict: dict) -> bool:
    return config_uses_mlir_backend(config_dict)


def _cfg_path_uses_mlir_backend(cfg_path: str) -> bool:
    return _config_dict_uses_mlir_backend(load_binding_config(cfg_path))


Config = NumbaConfig


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


def _mapping_values_to_names(values: dict[str, type]) -> dict[str, str]:
    return {key: value.__name__ for key, value in values.items()}


def _mlir_config_from_config(config: Config):
    from numbast.experimental.mlir.tools.static_binding_generator import (
        Config as MlirConfig,
    )

    config_dict = {
        "Entry Point": config.entry_point,
        "GPU Arch": config.gpu_arch,
        "File List": config.retain_list,
        "Types": _mapping_values_to_names(config.types),
        "Data Models": _mapping_values_to_names(config.datamodels),
        "Exclude": {
            "Function": config.exclude_functions,
            "Struct": config.exclude_structs,
        },
        "Clang Include Paths": config.clang_includes_paths,
        "Additional Import": config.additional_imports,
        "Shim Include Override": config.shim_include_override,
        "Predefined Macros": config.predefined_macros,
        "Output Name": config.output_name,
        "Cooperative Launch Required Functions Regex": (
            config.cooperative_launch_required_functions_regex
        ),
        "API Prefix Removal": config.api_prefix_removal,
        "Module Callbacks": config.module_callbacks,
        "Module Link Variables Used": config.module_link_variables_used,
        "Skip Prefix": config.skip_prefix,
        "Use Separate Registry": config.separate_registry,
        "Function Argument Intents": config.function_argument_intents,
    }
    return MlirConfig(config_dict)


def _run_mlir_static_binding_generator(
    config,
    output_dir: str,
    log_generates: bool = False,
    cfg_file_path: str | None = None,
    sbg_params: dict[str, str] = {},
    bypass_parse_error: bool = False,
) -> str:
    from numbast.experimental.mlir.static import (
        reset_renderer as reset_mlir_renderer,
    )
    from numbast.experimental.mlir.tools.static_binding_generator import (
        _static_binding_generator as mlir_static_binding_generator,
    )

    reset_mlir_renderer()
    return mlir_static_binding_generator(
        config,
        output_dir,
        log_generates=log_generates,
        cfg_file_path=cfg_file_path,
        sbg_params=sbg_params,
        bypass_parse_error=bypass_parse_error,
    )


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
    function_argument_intents: dict | None = None,
):
    """
    Render struct declarations into generated Python source for struct bindings.

    Parameters:
        struct_decls (list): Struct declaration objects to render.
        header_path (str): Path to the original header file associated with the declarations.
        types (dict): Mapping from struct name to the corresponding numba type object, if available.
        data_models (dict): Mapping from struct name to the corresponding numba datamodel object, if available.
        struct_prefix_removal (list): Name prefixes to remove from struct identifiers when rendering.
        excludes (list): Struct names to exclude from rendering.
        function_argument_intents (dict | None): Optional mapping describing argument intent metadata to influence rendering (defaults to an empty dict).

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
        function_argument_intents=function_argument_intents or {},
    )

    return SSR.render_as_str(with_imports=False, with_shim_stream=False)


def _generate_functions(
    func_decls: list[Function],
    header_path: str,
    excludes: list[str],
    cooperative_launch_functions: list[str],
    function_prefix_removal: list[str],
    skip_prefix: str | None,
    function_argument_intents: dict | None = None,
) -> str:
    """
    Render the function-binding source for the given function declarations.

    Parameters:
        func_decls (list[Function]): Parsed function declarations to render.
        header_path (str): Path to the original header file used for the shim stream.
        excludes (list[str]): Function names to exclude from rendering.
        cooperative_launch_functions (list[str]): Regex patterns or exact names identifying functions that require cooperative-launch handling.
        function_prefix_removal (list[str]): Prefixes to strip from function names when generating bindings.
        skip_prefix (str | None): If provided, skip generating bindings for functions whose names start with this prefix.
        function_argument_intents (dict | None): Mapping from function names to argument-intent specifications used to guide parameter handling during rendering.

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
        function_argument_intents=function_argument_intents or {},
    )

    return SFR.render_as_str(with_imports=False, with_shim_stream=False)


def _generate_function_templates(
    function_template_decls: list[FunctionTemplate],
    excludes: list[str],
    skip_prefix: str | None,
    function_argument_intents: dict | None = None,
) -> str:
    """
    Render static bindings for function templates.

    Parameters:
        function_template_decls (list[FunctionTemplate]): Parsed function-template declarations to render.
        excludes (list[str]): Function-template names to exclude.
        skip_prefix (str | None): Optional prefix used to skip function-template names.
        function_argument_intents (dict | None): Optional argument-intent overrides.

    Returns:
        str: Generated source code for function-template bindings.
    """
    SFTR = StaticFunctionTemplatesRenderer(
        function_template_decls,
        excludes=excludes,
        skip_prefix=skip_prefix,
        skip_non_device=True,
        function_argument_intents=function_argument_intents or {},
    )
    return SFTR.render_as_str(with_imports=False, with_shim_stream=False)


def _generate_class_templates(
    class_template_decls: list[ClassTemplate],
    header_path: str,
    excludes: list[str],
    function_argument_intents: dict | None = None,
) -> str:
    """
    Render static bindings for class templates.

    Parameters:
        class_template_decls (list[ClassTemplate]): Parsed class-template declarations to render.
        header_path (str): Header path used for template specialization parsing at runtime.
        excludes (list[str]): Class-template names (short or qualified) to exclude.
        function_argument_intents (dict | None): Optional argument-intent overrides.

    Returns:
        str: Generated source code for class-template bindings.
    """
    SCTR = StaticClassTemplatesRenderer(
        class_template_decls,
        header_path=header_path,
        excludes=excludes,
        function_argument_intents=function_argument_intents or {},
    )
    return SCTR.render_as_str(with_imports=False, with_shim_stream=False)


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
    function_templates: list[FunctionTemplate],
    structs: list[Struct],
    class_templates: list[ClassTemplate],
    enums: list[Enum],
    typedefs: list[Typedef],
):
    """Console log the list of bindings to generate."""

    click.echo("-" * 80)
    click.echo(
        "Generating bindings for "
        f"{len(functions)} functions, "
        f"{len(function_templates)} function templates, "
        f"{len(structs)} structs, "
        f"{len(class_templates)} class templates, "
        f"{len(typedefs)} typedefs, "
        f"{len(enums)} enums."
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
    click.echo("Function Templates: ")
    click.echo(
        "\n".join(
            f"  - {templ.function.qual_name}" for templ in function_templates
        )
    )
    click.echo("\nStructs: ")
    click.echo("\n".join(f"  - {struct.name}" for struct in structs))
    click.echo("Class Templates: ")
    click.echo(
        "\n".join(f"  - {templ.record.qual_name}" for templ in class_templates)
    )


def _static_binding_generator(
    config: Config,
    output_dir: str,
    log_generates: bool = False,
    cfg_file_path: str | None = None,
    sbg_params: dict[str, str] = {},
    bypass_parse_error: bool = False,
) -> str:
    """
    Generate static Python bindings for a CUDA C++ header according to the provided configuration.

    Parameters:
        cfg_file_path (str | None): Path to the YAML config used to produce these bindings; used for reproducible metadata and may be None.
        sbg_params (dict[str, str]): Additional key/value parameters to embed in the generator metadata.
        bypass_parse_error (bool): If True, continue generation when source parsing reports recoverable errors.

    Returns:
        str: Absolute path to the generated binding file.
    """
    if config.mlir_backend:
        mlir_config = _mlir_config_from_config(config)
        return _run_mlir_static_binding_generator(
            mlir_config,
            output_dir,
            log_generates=log_generates,
            cfg_file_path=cfg_file_path,
            sbg_params=sbg_params,
            bypass_parse_error=bypass_parse_error,
        )

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
    function_templates = decls.function_templates
    enums = decls.enums
    class_templates = decls.class_templates
    typedefs = [
        td
        for td in decls.typedefs
        if td.underlying_name not in config.exclude_structs
    ]

    if log_generates:
        log_files_to_generate(
            functions,
            function_templates,
            structs,
            class_templates,
            enums,
            typedefs,
        )

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
        config.function_argument_intents,
    )

    function_bindings = _generate_functions(
        functions,
        entry_point,
        config.exclude_functions,
        config.cooperative_launch_required_functions_regex,
        config.api_prefix_removal.get("Function", []),
        config.skip_prefix,
        config.function_argument_intents,
    )
    class_template_bindings = _generate_class_templates(
        class_templates,
        entry_point,
        config.exclude_structs,
        config.function_argument_intents,
    )
    function_template_bindings = _generate_function_templates(
        function_templates,
        config.exclude_functions,
        config.skip_prefix,
        config.function_argument_intents,
    )

    if config.separate_registry and (function_templates or class_templates):
        warnings.warn(
            "Function/class template static bindings currently register into "
            "Numba's default CUDA registries. "
            "'Use Separate Registry' does not yet isolate template bindings."
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
    callconv_utils_str = get_callconv_utils()
    imports_str = get_rendered_imports(
        additional_imports=config.additional_imports
    )

    # Example: Save the processed output to the output directory
    if config.output_name is None:
        output_file = os.path.join(output_dir, f"{basename}.py")
    else:
        output_file = os.path.join(output_dir, config.output_name)

    # Full command line that generated the binding:
    #
    # NOTE: This generator is frequently invoked from within other Python
    # programs (e.g. pytest via click's CliRunner). In such cases, `sys.argv`
    # reflects the *outer* process (pytest) rather than the generator itself,
    # which makes the embedded metadata unstable and can lead to confusing
    # output (and even false positives in tests that inspect the generated
    # bindings).
    if cfg_file_path is not None:
        cmd = (
            f"static_binding_generator --cfg-path {cfg_file_path} "
            f"--output-dir {output_dir}"
        )
    else:
        cmd = "<programmatic invocation>"

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
{callconv_utils_str}
# Enums:
{enum_bindings}
# Structs:
{struct_bindings}
# Functions:
{function_bindings}
# Class Templates:
{class_template_bindings}
# Function Templates:
{function_template_bindings}
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
    if _cfg_path_uses_mlir_backend(cfg_path):
        from numbast.experimental.mlir.tools.static_binding_generator import (
            Config as MlirConfig,
        )

        cfg = MlirConfig.from_yaml_path(cfg_path)
        output_file = _run_mlir_static_binding_generator(
            cfg,
            output_dir,
            log_generates=True,
            cfg_file_path=cfg_path,
            sbg_params=ctx.params,
            bypass_parse_error=bypass_parse_error,
        )
    else:
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
