# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Numba Python binding generation."""

from __future__ import annotations

import re

from numba import types
from numba.core.datamodel import models

from numbast.tools.config.shared import (
    SharedConfig,
    config_uses_mlir_backend,
)


_MLIR_BACKEND_ONLY_CONFIG_KEYS = ("Module Link Variables Used",)


def _config_value_is_set(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return True


def _validate_mlir_backend_only_config(config_dict: dict):
    if config_uses_mlir_backend(config_dict):
        return

    keys = [
        key
        for key in _MLIR_BACKEND_ONLY_CONFIG_KEYS
        if _config_value_is_set(config_dict.get(key))
    ]
    if not keys:
        return

    if len(keys) == 1:
        message = f'Configuration option "{keys[0]}" requires'
    else:
        options = ", ".join(f'"{key}"' for key in keys)
        message = f"Configuration options {options} require"

    raise ValueError(f'{message} "MLIR Backend: true".')


def _str_value_to_numba_type(values: dict[str, str]) -> dict[str, type]:
    """Convert string type names to Numba type objects."""

    return {key: getattr(types, value) for key, value in values.items()}


def _str_value_to_numba_datamodel(
    values: dict[str, str],
) -> dict[str, type]:
    """Convert string model names to Numba data model objects."""

    return {key: getattr(models, value) for key, value in values.items()}


class NumbaConfig(SharedConfig):
    """Configuration for Numba Python binding generation."""

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
    module_link_variables_used: list[str]
    skip_prefix: str | None
    separate_registry: bool
    function_argument_intents: dict
    mlir_backend: bool

    def __init__(self, config_dict: dict):
        self.mlir_backend = config_uses_mlir_backend(config_dict)
        _validate_mlir_backend_only_config(config_dict)
        super().__init__(config_dict)
        self.types = _str_value_to_numba_type(config_dict.get("Types", {}))
        self.datamodels = _str_value_to_numba_datamodel(
            config_dict.get("Data Models", {})
        )

        self.additional_imports = config_dict.get("Additional Import", [])
        self.shim_include_override = config_dict.get(
            "Shim Include Override", None
        )
        self.output_name = config_dict.get("Output Name", None)
        self.cooperative_launch_required_functions_regex = config_dict.get(
            "Cooperative Launch Required Functions Regex", []
        )
        self.module_callbacks = config_dict.get("Module Callbacks", {})
        self.module_link_variables_used = (
            config_dict.get("Module Link Variables Used", []) or []
        )
        self.separate_registry = config_dict.get("Use Separate Registry", False)
        self.function_argument_intents = (
            config_dict.get("Function Argument Intents", {}) or {}
        )

        self._verify_regex_patterns()

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
        module_link_variables_used: list[str] | None = None,
        skip_prefix: str | None = None,
        separate_registry: bool = False,
        function_argument_intents: dict | None = None,
        mlir_backend: bool = False,
    ) -> NumbaConfig:
        """Construct configuration from explicit parameters."""

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
            "Module Link Variables Used": module_link_variables_used or [],
            "Skip Prefix": skip_prefix,
            "Use Separate Registry": separate_registry,
            "Function Argument Intents": function_argument_intents or {},
            "MLIR Backend": mlir_backend,
        }

        if types:
            config_dict["Types"] = {
                key: value.__name__ for key, value in types.items()
            }
        if datamodels:
            config_dict["Data Models"] = {
                key: value.__name__ for key, value in datamodels.items()
            }

        return cls(config_dict)

    def _verify_regex_patterns(self):
        for pattern in self.cooperative_launch_required_functions_regex:
            try:
                re.compile(pattern)
            except re.error:
                raise ValueError(f"Invalid regex pattern: {pattern}")
