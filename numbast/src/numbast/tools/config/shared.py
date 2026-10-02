# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by Numbast binding generators."""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

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


def config_uses_mlir_backend(config_dict: dict) -> bool:
    """Return whether the Python bindings use the MLIR implementation."""

    return bool(
        config_dict.get(
            "MLIR Backend",
            config_dict.get("mlir_backend", False),
        )
    )


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


class SharedConfig:
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
