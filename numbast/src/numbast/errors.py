# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class BaseASTError(Exception):
    pass


class TypeNotFoundError(BaseASTError):
    """Indicate that a type string was not found in Numbast's type cache.

    Numbast adopts an incremental binding-building strategy. Libraries that are
    not self-contained may interoperate with declarations from other libraries.
    If third-party declarations do not exist in Numbast's type cache, these
    interoperations are ignored when building bindings.
    """

    def __init__(self, type_name):
        self._type_name = type_name
        super().__init__(f"{type_name} is not found in type cache.")

    @property
    def type_name(self):
        return self._type_name


class MangledFunctionNameConflictError(BaseASTError):
    """Indicate that a mangled function name is not unique.

    This error is raised when a function shares a mangled name with a previously
    generated binding.
    """

    def __init__(self, mangled_name: str):
        self._mangled_name = mangled_name
        super().__init__(f"Mangled function name {mangled_name} is not unique.")

    @property
    def mangled_name(self):
        return self._mangled_name


class CudaOxideBindingError(ValueError):
    """Raised after collecting every actionable CUDA-Oxide diagnostic."""

    def __init__(self, diagnostics: list[str]):
        self.diagnostics = sorted(set(diagnostics))
        details = "\n".join(f"  - {item}" for item in self.diagnostics)
        super().__init__(f"CUDA-Oxide binding generation failed:\n{details}")
