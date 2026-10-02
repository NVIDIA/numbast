# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for CUDA-Oxide Rust binding generation."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from numbast.rust_types import cuda_arch_number
from numbast.tools.config.common import SharedConfig, _as_list


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


class CudaOxideConfig(SharedConfig):
    """Configuration for CUDA-Oxide Rust binding generation."""

    def __init__(self, config_dict: dict[str, Any]):
        super().__init__(config_dict)

        section = config_dict.get("CUDA Oxide")
        if not isinstance(section, dict):
            raise TypeError('Configuration option "CUDA Oxide" must be a mapping.')
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
                '"CUDA Oxide.Manifest Name" and "CUDA Oxide.Output Name" must differ.'
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
                raise ValueError('"CUDA Oxide.Symbol Inventory" must be a file path.')
            if not os.path.isfile(self.symbol_inventory):
                raise ValueError(
                    f"Symbol inventory file does not exist: {self.symbol_inventory}"
                )

        self.bypass_parse_error = _as_bool(
            section.get("Bypass Parse Errors", False),
            "CUDA Oxide.Bypass Parse Errors",
        )
        self.clang_binary = section.get("Clang Binary")
        if self.clang_binary is not None and not isinstance(self.clang_binary, str):
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
