# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module
from importlib import metadata

try:
    __version__ = metadata.version("numbast")
except metadata.PackageNotFoundError:
    __version__ = "0+unknown"

_LAZY_ATTRS = {
    "bind_cxx_enum": ("numbast.enum", "bind_cxx_enum"),
    "bind_cxx_enums": ("numbast.enum", "bind_cxx_enums"),
    "bind_cxx_function": ("numbast.function", "bind_cxx_function"),
    "bind_cxx_functions": ("numbast.function", "bind_cxx_functions"),
    "bind_cxx_function_template": (
        "numbast.function_template",
        "bind_cxx_function_template",
    ),
    "bind_cxx_function_templates": (
        "numbast.function_template",
        "bind_cxx_function_templates",
    ),
    "bind_cxx_struct": ("numbast.struct", "bind_cxx_struct"),
    "bind_cxx_structs": ("numbast.struct", "bind_cxx_structs"),
    "bind_cxx_class_template_specialization": (
        "numbast.class_template",
        "bind_cxx_class_template_specialization",
    ),
    "bind_cxx_class_template": (
        "numbast.class_template",
        "bind_cxx_class_template",
    ),
    "bind_cxx_class_templates": (
        "numbast.class_template",
        "bind_cxx_class_templates",
    ),
    "clear_concrete_type_caches": (
        "numbast.class_template",
        "clear_concrete_type_caches",
    ),
    "MemoryShimWriter": ("numbast.shim_writer", "MemoryShimWriter"),
    "FileShimWriter": ("numbast.shim_writer", "FileShimWriter"),
}

_numba_version_checked = False


def _ensure_supported_numba() -> None:
    global _numba_version_checked
    if _numba_version_checked:
        return

    import numba

    _major, minor, *_ = numba.__version__.split(".")
    if int(minor) < 59:
        raise RuntimeError("Numba version >= 0.59rc1 is required")

    _numba_version_checked = True


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _ensure_supported_numba()
    module_name, attr_name = _LAZY_ATTRS[name]
    attr = getattr(import_module(module_name), attr_name)
    globals()[name] = attr
    return attr


def __dir__():
    return sorted({*globals(), *_LAZY_ATTRS})


__all__ = [
    "__version__",
    "bind_cxx_enum",
    "bind_cxx_enums",
    "bind_cxx_function",
    "bind_cxx_functions",
    "bind_cxx_function_template",
    "bind_cxx_function_templates",
    "bind_cxx_struct",
    "bind_cxx_structs",
    "bind_cxx_class_template_specialization",
    "bind_cxx_class_template",
    "bind_cxx_class_templates",
    "clear_concrete_type_caches",
    "MemoryShimWriter",
    "FileShimWriter",
]
