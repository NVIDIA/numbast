# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate direct CUDA-Oxide device externs for a C-linkage CUDA API."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import click

from numbast.binding_model import (
    CUDA_ABI_ALIASES,
    AbiTypedef,
    BindingModel,
    BindingModelError,
    build_c_device_model,
    cuda_abi_alias_for_arch,
    modern_nvvm_required_symbols,
    parse_c_type_spelling,
    rust_type,
    translate_constant_literal,
)
from numbast.name_policy import is_identifier, rust_identifier
from numbast.tools.binding_config import CudaOxideConfig

MANIFEST_SCHEMA_VERSION = 1
_RUST_MAX_WIDTH = 100


def _package_version(package: str) -> str:
    source_root = Path(__file__).resolve().parents[4]
    version_file = source_root / "VERSION"
    if version_file.is_file() and (
        package == "numbast" or (source_root / package).is_dir()
    ):
        return f"{version_file.read_text(encoding='utf-8').strip()}+source"
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "0+unknown"


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        chunk = source.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = source.read(1024 * 1024)
    return digest.hexdigest()


def _cuda_toolkit_version() -> str | None:
    candidates = []
    for name in ("CUDA_PATH", "CUDA_HOME"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    visited = set()
    for candidate in candidates:
        for root in (candidate, *list(candidate.parents)[:3]):
            if root in visited:
                continue
            visited.add(root)
            version_json = root / "version.json"
            if version_json.is_file():
                try:
                    contents = json.loads(
                        version_json.read_text(encoding="utf-8")
                    )
                    version = contents["cuda"]["version"]
                except (KeyError, OSError, TypeError, ValueError):
                    pass
                else:
                    return str(version)
            version_text = root / "version.txt"
            if version_text.is_file():
                try:
                    return version_text.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
    return None


def _write_text_atomic(path: Path, contents: str):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary.write(contents)
            temporary_path = temporary.name
        # NamedTemporaryFile defaults to owner-only permissions. Generated
        # source and manifests are ordinary build artifacts and should remain
        # readable when copied into a package or checked into source control.
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _add_supplemental_typedefs(model: BindingModel, config: CudaOxideConfig):
    occupied = {
        *(item.name for item in model.enums if item.name),
        *(item.name for item in model.records),
        *(item.name for item in model.typedefs),
        *model.cuda_aliases,
    }
    diagnostics = []
    for name, spelling in sorted(config.type_aliases.items()):
        if not is_identifier(name):
            diagnostics.append(
                f"supplemental type alias has invalid name {name!r}"
            )
            continue
        try:
            underlying = parse_c_type_spelling(spelling)
        except ValueError as error:
            diagnostics.append(f"supplemental type alias {name!r}: {error}")
            continue
        existing = next(
            (item for item in model.typedefs if item.name == name), None
        )
        if existing is not None and existing.underlying == underlying:
            continue
        if name in occupied:
            diagnostics.append(
                f"supplemental type alias {name!r} conflicts with a parsed type"
            )
            continue
        model.typedefs.append(AbiTypedef(name, underlying))
        if underlying.base_name in CUDA_ABI_ALIASES:
            model.cuda_aliases[underlying.base_name] = cuda_abi_alias_for_arch(
                underlying.base_name, config.gpu_arch[0]
            )
        occupied.add(name)

    # Validate after all aliases are installed so forward alias references work.
    aliases = {item.name: item.underlying.base_name for item in model.typedefs}

    def visit(name: str, path: tuple[str, ...]):
        if name in path:
            cycle = " -> ".join((*path[path.index(name) :], name))
            diagnostics.append(f"cyclic type aliases: {cycle}")
            return
        target = aliases.get(name)
        if target in aliases:
            visit(target, (*path, name))

    for name in sorted(aliases):
        visit(name, ())

    for alias in model.typedefs:
        try:
            rust_type(alias.underlying, model)
        except ValueError as error:
            diagnostics.append(f"type alias {alias.name!r}: {error}")

    rendered_names: dict[str, str] = {}
    type_names = [
        *(item.name for item in model.enums if item.name),
        *(item.name for item in model.records),
        *(item.name for item in model.typedefs),
        *model.cuda_aliases,
    ]
    for name in sorted(type_names):
        rendered_name = rust_identifier(name)
        previous = rendered_names.get(rendered_name)
        if previous is not None and previous != name:
            diagnostics.append(
                f"Rust type name collision: {previous!r} and {name!r} both "
                f"map to {rendered_name!r}"
            )
        rendered_names[rendered_name] = name
    model.typedefs.sort(key=lambda item: item.name)
    if diagnostics:
        raise BindingModelError(diagnostics)


def _render_constants(model: BindingModel, config: CudaOxideConfig):
    lines = []
    rendered = []
    occupied: dict[str, tuple[str, tuple[str, str]]] = {}

    for enum in model.enums:
        enum_type = (
            rust_identifier(enum.name)
            if enum.name
            else enum.rust_underlying_type
        )
        for name, value in enum.enumerators:
            if not is_identifier(name):
                raise BindingModelError(
                    [f"enum constant has invalid name {name!r}"]
                )
            rust_constant_name = rust_identifier(name)
            item = (enum_type, value)
            previous = occupied.get(rust_constant_name)
            if previous is not None:
                previous_name, previous_item = previous
                if previous_name == name and previous_item == item:
                    continue
                raise BindingModelError(
                    [
                        f"constant name collision: {previous_name!r} and {name!r} "
                        f"both map to {rust_constant_name!r}"
                    ]
                )
            occupied[rust_constant_name] = (name, item)
            lines.append(
                f"pub const {rust_constant_name}: {enum_type} = {value};"
            )
            rendered.append(
                {
                    "name": name,
                    "rust_type": enum_type,
                    "value": value,
                    "source": "enum",
                }
            )

    for name, spec in sorted(config.constants.items()):
        if not is_identifier(name):
            raise BindingModelError(
                [f"supplemental constant has invalid name {name!r}"]
            )
        if not isinstance(spec, dict) or set(spec) != {"Type", "Value"}:
            raise BindingModelError(
                [
                    (
                        f"supplemental constant {name!r} must contain exactly "
                        '"Type" and "Value"'
                    )
                ]
            )
        if not isinstance(spec["Type"], str):
            raise BindingModelError(
                [
                    f"supplemental constant {name!r}: Type must be a C type spelling"
                ]
            )
        try:
            c_type = parse_c_type_spelling(spec["Type"])
            if (
                c_type.pointer_depth
                or c_type.array_dimensions
                or c_type.base_name == "void"
            ):
                raise ValueError(
                    "constant type must be a non-void scalar or alias"
                )
            rust_name = rust_type(c_type, model)
            value = translate_constant_literal(spec["Value"])
        except (TypeError, ValueError) as error:
            raise BindingModelError(
                [f"supplemental constant {name!r}: {error}"]
            )
        rust_constant_name = rust_identifier(name)
        if rust_constant_name in occupied:
            previous_name = occupied[rust_constant_name][0]
            raise BindingModelError(
                [
                    f"supplemental constant {name!r} conflicts with parsed "
                    f"constant {previous_name!r}"
                ]
            )
        occupied[rust_constant_name] = (name, (rust_name, value))
        lines.append(f"pub const {rust_constant_name}: {rust_name} = {value};")
        rendered.append(
            {
                "name": name,
                "rust_type": rust_name,
                "value": value,
                "source": "configuration",
            }
        )
    return lines, rendered


def _render_extern_function(function, model: BindingModel) -> list[str]:
    parameters = [
        f"{parameter.rust_name}: {rust_type(parameter.type_, model)}"
        for parameter in function.parameters
    ]
    result = ""
    if not (
        function.return_type.base_name == "void"
        and not function.return_type.pointer_depth
        and not function.return_type.array_dimensions
    ):
        result = f" -> {rust_type(function.return_type, model)}"

    name = rust_identifier(function.native_name)
    single_line = f"    pub fn {name}({', '.join(parameters)}){result};"
    if len(single_line) < _RUST_MAX_WIDTH:
        return [single_line]
    if len(single_line) == _RUST_MAX_WIDTH:
        if not result:
            return [single_line]
        signature = f"    pub fn {name}({', '.join(parameters)})"
        return [signature, f"        {result.strip()};"]
    return [
        f"    pub fn {name}(",
        *(f"        {parameter}," for parameter in parameters),
        f"    ){result};",
    ]


def render_cuda_oxide_bindings(
    model: BindingModel, config: CudaOxideConfig, config_path: str | None = None
) -> tuple[str, list[dict[str, str]]]:
    """Render an includable Rust module with no C++ shim layer."""

    _add_supplemental_typedefs(model, config)
    constant_lines, rendered_constants = _render_constants(model, config)
    lines = [
        "// SPDX-License-Identifier: Apache-2.0",
        "// Automatically generated by Numbast; do not edit.",
        f"// Source config: {config_path or '<programmatic>'}",
        "// Raw declarations are unsafe and resolve directly from external CUDA LTOIR.",
        "",
        "#[allow(unused_imports)]",
        "use cuda_device::device;",
        "",
    ]
    modern_only_symbols = modern_nvvm_required_symbols(model)
    architecture = config.gpu_arch_number
    if modern_only_symbols and architecture < 100:
        lines.extend(
            [
                (
                    f"// Compatibility: {len(modern_only_symbols)} declaration(s) "
                    "pass sub-32-bit values by value."
                ),
                "// CUDA-Oxide can call those declarations only on sm_100+; see the manifest.",
                "",
            ]
        )

    for name, (storage, size, alignment) in sorted(model.cuda_aliases.items()):
        rust_name = rust_identifier(name)
        lines.extend(
            [
                f"/// CUDA ABI storage: size {size}, alignment {alignment}.",
                "#[allow(non_camel_case_types)]",
                f"pub type {rust_name} = {storage};",
                f"const _: [(); {size}] = [(); core::mem::size_of::<{rust_name}>()];",
                f"const _: [(); {alignment}] = [(); core::mem::align_of::<{rust_name}>()];",
                "",
            ]
        )
    for record in model.records:
        rust_name = rust_identifier(record.name)
        lines.extend(
            [
                "/// Opaque POD storage used behind a device-extern pointer.",
                f"/// C layout: size {record.size}, alignment {record.alignment}.",
                "#[allow(non_camel_case_types)]",
                f"pub type {rust_name} = {record.storage_type};",
                f"const _: [(); {record.size}] = [(); core::mem::size_of::<{rust_name}>()];",
                f"const _: [(); {record.alignment}] = [(); core::mem::align_of::<{rust_name}>()];",
                "",
            ]
        )
    for enum in model.enums:
        if not enum.name:
            continue
        lines.extend(
            [
                "#[allow(non_camel_case_types)]",
                f"pub type {rust_identifier(enum.name)} = {enum.rust_underlying_type};",
                "",
            ]
        )
    for typedef in model.typedefs:
        if typedef.name in model.cuda_aliases:
            continue
        lines.extend(
            [
                "#[allow(non_camel_case_types)]",
                (
                    f"pub type {rust_identifier(typedef.name)} = "
                    f"{rust_type(typedef.underlying, model)};"
                ),
                "",
            ]
        )
    if constant_lines:
        lines.extend(["#[allow(non_upper_case_globals)]"])
        lines.extend(constant_lines)
        lines.append("")

    value_names = {rust_identifier(item["name"]) for item in rendered_constants}
    for function in model.functions:
        if (
            rust_identifier(function.public_name) in value_names
            or rust_identifier(function.native_name) in value_names
        ):
            raise BindingModelError(
                [
                    f"public function name {function.public_name!r} conflicts with a constant"
                ]
            )

    lines.extend(
        ["#[device]", "#[allow(improper_ctypes)]", 'unsafe extern "C" {']
    )
    for function in model.functions:
        lines.extend(_render_extern_function(function, model))
    lines.extend(["}", ""])

    aliases = [
        function
        for function in model.functions
        if function.public_name != function.native_name
    ]
    if aliases:
        lines.append(
            "// Prefix-stripped public names preserve the native link symbol above."
        )
        for function in aliases:
            lines.append(
                f"pub use self::{rust_identifier(function.native_name)} as "
                f"{rust_identifier(function.public_name)};"
            )
        lines.append("")
    return "\n".join(lines), rendered_constants


def _read_symbol_inventory(path: str) -> set[str]:
    symbols = set()
    diagnostics = []
    with open(path, encoding="utf-8") as inventory:
        for line_number, raw_line in enumerate(inventory, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = line.split()[-1]
            if is_identifier(candidate):
                symbols.add(candidate)
            else:
                diagnostics.append(
                    f"symbol inventory line {line_number} has invalid C symbol "
                    f"{candidate!r}"
                )
    if diagnostics:
        raise BindingModelError(diagnostics)
    return symbols


def verify_symbol_inventory(model: BindingModel, path: str) -> dict[str, Any]:
    """Require exact parity with a curated nm-style API symbol inventory."""

    available = _read_symbol_inventory(path)
    expected = {function.native_name for function in model.functions}
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    diagnostics = []
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        diagnostics.append(
            f"LTOIR symbol inventory is missing {len(missing)} generated "
            f"symbol(s): {preview}{suffix}"
        )
    if unexpected:
        preview = ", ".join(unexpected[:20])
        suffix = " ..." if len(unexpected) > 20 else ""
        diagnostics.append(
            f"LTOIR symbol inventory contains {len(unexpected)} ungenerated "
            f"symbol(s): {preview}{suffix}; provide the selected public API inventory"
        )
    if diagnostics:
        raise BindingModelError(diagnostics)
    return {
        "status": "verified",
        "path": path,
        "sha256": _sha256(path),
        "required_symbol_count": len(expected),
        "available_symbol_count": len(available),
        "missing_symbols": [],
        "unexpected_symbols": [],
    }


def _type_manifest(model: BindingModel) -> dict[str, Any]:
    return {
        "cuda_abi_aliases": [
            {
                "name": name,
                "rust_type": values[0],
                "size": values[1],
                "alignment": values[2],
            }
            for name, values in sorted(model.cuda_aliases.items())
        ],
        "enums": [asdict(item) for item in model.enums],
        "records": [asdict(item) for item in model.records],
        "typedefs": [
            {"name": item.name, "underlying": item.underlying.manifest_dict()}
            for item in model.typedefs
        ],
    }


def make_manifest(
    model: BindingModel,
    config: CudaOxideConfig,
    rust_output: str,
    constants: list[dict[str, str]],
    config_path: str | None = None,
) -> dict[str, Any]:
    symbols = []
    for function in model.functions:
        symbols.append(
            {
                "native_name": function.native_name,
                "public_name": function.public_name,
                "rust_name": rust_identifier(function.public_name),
                "execution_space": function.execution_space,
                "return_type": {
                    "c": function.return_type.manifest_dict(),
                    "rust": rust_type(function.return_type, model),
                },
                "parameters": [
                    {
                        "c_name": parameter.c_name,
                        "rust_name": parameter.rust_name,
                        "c_type": parameter.type_.manifest_dict(),
                        "rust_type": rust_type(parameter.type_, model),
                    }
                    for parameter in function.parameters
                ],
            }
        )

    source_paths = sorted({config.entry_point, *config.retain_list})
    modern_only_symbols = modern_nvvm_required_symbols(model)
    architecture = config.gpu_arch_number
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "backend": "cuda-oxide",
        "library": {"name": config.name, "version": config.version},
        "generator": {
            "name": "numbast-cuda-oxide",
            "numbast_version": _package_version("numbast"),
            "ast_canopy_version": _package_version("ast_canopy"),
            "cuda_toolkit_version": _cuda_toolkit_version(),
            "generation_command": (
                [
                    "numbast-cuda-oxide",
                    "--cfg-path",
                    config_path,
                    "--output-dir",
                    str(Path(rust_output).parent),
                ]
                if config_path is not None
                else None
            ),
        },
        "target": {"gpu_arch": config.gpu_arch[0], "abi": "nvptx64-c"},
        "compatibility": {
            "cuda_oxide_nvvm_dialect": (
                "modern-opaque-pointer"
                if architecture >= 100
                else "legacy-llvm-7"
            ),
            "modern_nvvm_required_symbols": modern_only_symbols,
            "selected_arch_supports_all_symbols": (
                architecture >= 100 or not modern_only_symbols
            ),
        },
        "source": {
            "config": config_path,
            "config_sha256": (
                _sha256(config_path)
                if config_path is not None and os.path.isfile(config_path)
                else None
            ),
            "entry_point": config.entry_point,
            "retained_files": list(config.retain_list),
            "clang_include_paths": list(config.clang_includes_paths),
            "predefined_macros": list(config.parser_defines),
            "parser": {
                "bypass_parse_errors": config.bypass_parse_error,
                "clang_binary": config.clang_binary,
            },
            "headers": [
                {"path": path, "sha256": _sha256(path)} for path in source_paths
            ],
        },
        "artifacts": {
            "rust_module": rust_output,
            "ltoir_inputs": list(config.ltoir_inputs),
            "ltoir": [
                {
                    "path": path,
                    **(
                        {"status": "present", "sha256": _sha256(path)}
                        if os.path.isfile(path)
                        else {"status": "not-found-at-generation"}
                    ),
                }
                for path in config.ltoir_inputs
            ],
        },
        "symbols": symbols,
        "types": _type_manifest(model),
        "constants": constants,
        "excluded_declarations": model.exclusions,
        "counts": {
            "generated_symbols": len(symbols),
            "excluded_declarations": len(model.exclusions),
        },
    }
    if config.symbol_inventory:
        manifest["symbol_verification"] = verify_symbol_inventory(
            model, config.symbol_inventory
        )
    else:
        manifest["symbol_verification"] = {
            "status": "not-requested",
            "hint": "Set CUDA Oxide.Symbol Inventory to an nm-style LTOIR symbol list.",
        }
    return manifest


def generate_cuda_oxide_bindings(
    config: CudaOxideConfig,
    output_dir: str | os.PathLike[str],
    config_path: str | None = None,
    declarations: Any | None = None,
) -> tuple[str, str]:
    """Parse, model, render, and write Rust bindings plus their manifest."""

    if declarations is None:
        from ast_canopy import parse_declarations_from_source

        declarations = parse_declarations_from_source(
            os.path.abspath(config.entry_point),
            [os.path.abspath(path) for path in config.retain_list],
            compute_capability=config.gpu_arch[0],
            additional_includes=config.clang_includes_paths,
            defines=config.parser_defines,
            bypass_parse_error=config.bypass_parse_error,
            clang_binary=config.clang_binary,
        )

    model = build_c_device_model(declarations, config)
    rendered, constants = render_cuda_oxide_bindings(model, config, config_path)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_name = config.output_name or f"{Path(config.entry_point).stem}.rs"
    rust_path = output_directory / output_name
    manifest_path = output_directory / config.manifest_name
    manifest = make_manifest(
        model, config, str(rust_path), constants, config_path=config_path
    )

    _write_text_atomic(rust_path, rendered)
    _write_text_atomic(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return str(rust_path), str(manifest_path)


@click.command()
@click.option(
    "--cfg-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, writable=True),
)
def cuda_oxide_binding_generator(cfg_path: str, output_dir: str):
    """Generate direct Rust device bindings and a versioned link manifest."""

    try:
        config = CudaOxideConfig.from_yaml_path(cfg_path)
        rust_path, manifest_path = generate_cuda_oxide_bindings(
            config, output_dir, config_path=cfg_path
        )
    except (
        BindingModelError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Generated CUDA-Oxide bindings: {rust_path}")
    click.echo(f"Generated CUDA-Oxide manifest: {manifest_path}")


if __name__ == "__main__":
    cuda_oxide_binding_generator()
