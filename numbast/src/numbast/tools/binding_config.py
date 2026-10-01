# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for Numbast binding generators.

A document may contain settings for multiple backends. Each backend reads the
same frontend configuration but renders independently in its own invocation.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from numbast.rust_types import cuda_arch_number
from numbast.tools.yaml_tags import string_constructor


class BindingConfigLoader(yaml.SafeLoader):
    """Safe YAML loader with Numbast's supported custom tags."""


BindingConfigLoader.add_constructor("!numbast_join", string_constructor)


def load_binding_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a Numbast YAML configuration without importing a backend."""

    with open(path, encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=BindingConfigLoader)
    if not isinstance(config, dict):
        raise TypeError("The binding configuration must be a YAML mapping.")
    return config


def _as_list(value: Any, key: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f'Configuration option "{key}" must be a list.')
    return list(value)


def _normalize_prefixes(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            'Configuration option "API Prefix Removal" must be a mapping.'
        )

    normalized = {}
    for kind, prefixes in value.items():
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if not isinstance(prefixes, list) or not all(
            isinstance(prefix, str) for prefix in prefixes
        ):
            raise ValueError(
                'Values in "API Prefix Removal" must be strings or lists of strings.'
            )
        normalized[kind] = list(prefixes)
    return normalized


def _as_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f'Configuration option "{key}" must be a boolean.')
    return value


def _as_filename(value: Any, key: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(f'Configuration option "{key}" must be a filename.')
    return value


class BindingConfig:
    """Configuration shared by all parser and binding backends."""

    def __init__(self, config_dict: dict[str, Any]):
        if not isinstance(config_dict, dict):
            raise TypeError("The binding configuration must be a YAML mapping.")

        missing = [
            key
            for key in ("Entry Point", "GPU Arch", "File List")
            if key not in config_dict
        ]
        if missing:
            raise ValueError(
                "Missing required configuration option(s): "
                + ", ".join(missing)
            )

        self.raw_config = copy.deepcopy(config_dict)
        self.name = config_dict.get("Name")
        self.version = config_dict.get("Version")
        self.entry_point = config_dict["Entry Point"]
        self.gpu_arch = _as_list(config_dict["GPU Arch"], "GPU Arch")
        self.retain_list = _as_list(config_dict["File List"], "File List")

        excludes = config_dict.get("Exclude", {}) or {}
        if not isinstance(excludes, dict):
            raise TypeError('Configuration option "Exclude" must be a mapping.')
        self.excludes = copy.deepcopy(excludes)
        self.exclude_functions = _as_list(
            self.excludes.get("Function", []), "Exclude.Function"
        )
        self.exclude_structs = _as_list(
            self.excludes.get("Struct", []), "Exclude.Struct"
        )

        self.clang_includes_paths = _as_list(
            config_dict.get("Clang Include Paths", []), "Clang Include Paths"
        )
        self.predefined_macros = _as_list(
            config_dict.get("Predefined Macros", []), "Predefined Macros"
        )
        self.api_prefix_removal = _normalize_prefixes(
            config_dict.get("API Prefix Removal", {})
        )
        self.skip_prefix = config_dict.get("Skip Prefix")

        if not isinstance(self.entry_point, str):
            raise TypeError(
                'Configuration option "Entry Point" must be a string.'
            )
        if not self.gpu_arch:
            raise ValueError("At least one GPU architecture must be provided.")
        if not all(isinstance(arch, str) for arch in self.gpu_arch):
            raise ValueError(
                'Configuration option "GPU Arch" must contain strings.'
            )
        if len(self.gpu_arch) > 1:
            raise NotImplementedError(
                "Multiple GPU architectures are not supported yet."
            )
        if not all(isinstance(path, str) for path in self.retain_list):
            raise ValueError(
                'Configuration option "File List" must contain strings.'
            )
        if self.skip_prefix is not None and not isinstance(
            self.skip_prefix, str
        ):
            raise ValueError(
                'Configuration option "Skip Prefix" must be a string.'
            )

        self._verify_exists()

    @classmethod
    def from_yaml_path(cls, cfg_path: str | os.PathLike[str]):
        """Create a backend configuration from a shared YAML document."""

        return cls(load_binding_config(cfg_path))

    def _verify_exists(self):
        if not os.path.exists(self.entry_point):
            raise ValueError(
                f"Input header file does not exist: {self.entry_point}"
            )
        for path in self.retain_list:
            if not os.path.exists(path):
                raise ValueError(f"File in retain list does not exist: {path}")
        for path in self.clang_includes_paths:
            if not os.path.exists(path):
                raise ValueError(f"File in include list does not exist: {path}")


class CudaOxideConfig(BindingConfig):
    """Configuration for the CUDA-Oxide C-device backend."""

    def __init__(self, config_dict: dict[str, Any]):
        super().__init__(config_dict)

        section = config_dict.get("CUDA Oxide")
        if not isinstance(section, dict):
            raise TypeError(
                'Configuration option "CUDA Oxide" must be a mapping.'
            )
        supported_options = {
            "Bypass Parse Errors",
            "Clang Binary",
            "Constants",
            "LTOIR Inputs",
            "Manifest Name",
            "Output Name",
            "Symbol Inventory",
            "Type Aliases",
        }
        unknown_options = sorted(set(section) - supported_options)
        if unknown_options:
            raise ValueError(
                'Unknown "CUDA Oxide" configuration option(s): '
                + ", ".join(unknown_options)
            )

        self.ltoir_inputs = _as_list(
            section.get("LTOIR Inputs", []), "CUDA Oxide.LTOIR Inputs"
        )
        if not self.ltoir_inputs or not all(
            isinstance(path, str) and path for path in self.ltoir_inputs
        ):
            raise ValueError(
                '"CUDA Oxide.LTOIR Inputs" must contain at least one path.'
            )

        default_output_name = f"{Path(self.entry_point).stem}.rs"
        self.output_name = _as_filename(
            section.get("Output Name", default_output_name),
            "CUDA Oxide.Output Name",
        )
        default_manifest_name = f"{Path(self.output_name).stem}.manifest.json"
        self.manifest_name = _as_filename(
            section.get("Manifest Name", default_manifest_name),
            "CUDA Oxide.Manifest Name",
        )
        if self.manifest_name == self.output_name:
            raise ValueError(
                '"CUDA Oxide.Manifest Name" and "CUDA Oxide.Output Name" '
                "must differ."
            )

        self.constants = copy.deepcopy(section.get("Constants", {}) or {})
        if not isinstance(self.constants, dict):
            raise TypeError(
                'Configuration option "CUDA Oxide.Constants" must be a mapping.'
            )
        self.type_aliases = copy.deepcopy(section.get("Type Aliases", {}) or {})
        if not isinstance(self.type_aliases, dict) or not all(
            isinstance(name, str)
            and isinstance(underlying, str)
            and name
            and underlying
            for name, underlying in self.type_aliases.items()
        ):
            raise ValueError(
                '"CUDA Oxide.Type Aliases" must map C names to C type spellings.'
            )

        self.symbol_inventory = section.get("Symbol Inventory")
        if self.symbol_inventory is not None:
            if not isinstance(self.symbol_inventory, str):
                raise ValueError(
                    '"CUDA Oxide.Symbol Inventory" must be a file path.'
                )
            if not os.path.isfile(self.symbol_inventory):
                raise ValueError(
                    f"Symbol inventory file does not exist: {self.symbol_inventory}"
                )

        self.bypass_parse_error = _as_bool(
            section.get("Bypass Parse Errors", False),
            "CUDA Oxide.Bypass Parse Errors",
        )
        self.clang_binary = section.get("Clang Binary")
        if self.clang_binary is not None and not isinstance(
            self.clang_binary, str
        ):
            raise TypeError(
                'Configuration option "CUDA Oxide.Clang Binary" must be a string.'
            )

        try:
            self.gpu_arch_number = cuda_arch_number(self.gpu_arch[0])
        except ValueError as error:
            raise ValueError(
                'CUDA-Oxide "GPU Arch" must use a supported sm_<digits> target.'
            ) from error
        expected_cuda_arch = str(self.gpu_arch_number * 10)
        cuda_arch_macros = [
            macro
            for macro in self.predefined_macros
            if macro == "__CUDA_ARCH__" or macro.startswith("__CUDA_ARCH__=")
        ]
        if len(cuda_arch_macros) > 1:
            raise ValueError(
                '"Predefined Macros" contains multiple __CUDA_ARCH__ definitions.'
            )
        if cuda_arch_macros:
            supplied = cuda_arch_macros[0].partition("=")[2]
            if supplied != expected_cuda_arch:
                raise ValueError(
                    f"__CUDA_ARCH__={supplied or '<empty>'} does not match "
                    f'"GPU Arch: {self.gpu_arch[0]}"; expected '
                    f"__CUDA_ARCH__={expected_cuda_arch}."
                )
            self.parser_defines = list(self.predefined_macros)
        else:
            self.parser_defines = [
                *self.predefined_macros,
                f"__CUDA_ARCH__={expected_cuda_arch}",
            ]
