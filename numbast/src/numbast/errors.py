# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class BaseASTError(Exception):
    pass


class TypeNotFoundError(BaseASTError):
    """Indicate that a type string is not found within Numbast's type cache.

    Numbast adopts an "incremental binding building" strategy. Libraries that are not
    self-contained can have inter-operations with declarations in other libraries.
    In these situations, if third party library decls do not pre-exist in Numbast's
    type cache, Numbast chooses to ignore these inter-operations when doing binding
    building.
    """

    def __init__(self, type_name):
        self._type_name = type_name
        super().__init__(f"{type_name} is not found in type cache.")

    @property
    def type_name(self):
        return self._type_name


class MangledFunctionNameConflictError(BaseASTError):
    """Indicate that a mangled function name is not unique.

    This error is raised when a function with the same mangled name of a previously
    generated binding is found.
    """

    def __init__(self, mangled_name: str):
        self._mangled_name = mangled_name
        super().__init__(f"Mangled function name {mangled_name} is not unique.")

    @property
    def mangled_name(self):
        return self._mangled_name
