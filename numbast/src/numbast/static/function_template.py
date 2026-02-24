# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import pickle

from ast_canopy.decl import FunctionTemplate
from ast_canopy.pylibastcanopy import execution_space

from numbast.function_template import (
    bind_cxx_function_templates,
    func_obj_registry as _function_template_registry,
)
from numbast.static.renderer import (
    BaseRenderer,
    get_callconv_utils,
    get_rendered_imports,
    get_shim,
)


def clear_function_template_registry():
    """Clear cached function-template Python handles used by dynamic binders."""
    _function_template_registry.clear()


def _decode_pickled_function_templates(
    encoded_function_templates: str,
) -> list[FunctionTemplate]:
    if not encoded_function_templates:
        return []
    raw = base64.b64decode(encoded_function_templates.encode("ascii"))
    return pickle.loads(raw)


def bind_static_function_templates(
    *,
    namespace: dict[str, object],
    encoded_function_templates: str,
    shim_writer,
    skip_prefix: str | None = None,
    skip_non_device: bool = True,
    exclude: list[str] | None = None,
    arg_intent: dict | None = None,
) -> list[object]:
    """
    Materialize and bind function templates into a generated static module namespace.

    Parameters
    ----------
    namespace : dict[str, object]
        Usually ``globals()`` of the generated binding module.
    encoded_function_templates : str
        Base64-encoded pickle payload containing ``list[FunctionTemplate]``.
    shim_writer : object
        Shim writer adapter exposed by the generated static binding script.
    skip_prefix : str | None
        Skip templates whose function name starts with this prefix.
    skip_non_device : bool
        Skip non-device functions.
    exclude : list[str] | None
        Function names to exclude from binding.
    arg_intent : dict | None
        Optional argument intent overrides.

    Returns
    -------
    list[object]
        Bound function-template handles.
    """
    function_templates = _decode_pickled_function_templates(
        encoded_function_templates
    )
    funcs = bind_cxx_function_templates(
        function_templates=function_templates,
        shim_writer=shim_writer,
        skip_prefix=skip_prefix,
        skip_non_device=skip_non_device,
        exclude=set(exclude or []),
        arg_intent=arg_intent,
    )

    for func in funcs:
        namespace[func.__name__] = func

    return funcs


class StaticFunctionTemplatesRenderer(BaseRenderer):
    """Render static bindings for function templates."""

    function_templates_binding_template = """
bind_static_function_templates(
    namespace=globals(),
    encoded_function_templates=\"\"\"{encoded_function_templates}\"\"\",
    shim_writer=shim_writer,
    skip_prefix={skip_prefix},
    skip_non_device={skip_non_device},
    exclude={exclude},
    arg_intent={arg_intent},
)
"""

    def __init__(
        self,
        function_templates: list[FunctionTemplate],
        *,
        skip_prefix: str | None = None,
        skip_non_device: bool = True,
        excludes: list[str] | None = None,
        function_argument_intents: dict | None = None,
    ):
        super().__init__(function_templates)
        self._function_templates = function_templates
        self._skip_prefix = skip_prefix
        self._skip_non_device = skip_non_device
        self._excludes = excludes or []
        self._function_argument_intents = function_argument_intents or {}

    def _is_eligible_template(self, templ: FunctionTemplate) -> bool:
        func_decl = templ.function
        if self._skip_non_device and func_decl.exec_space not in {
            execution_space.device,
            execution_space.host_device,
        }:
            return False

        if self._skip_prefix and func_decl.name.startswith(self._skip_prefix):
            return False

        if func_decl.name in self._excludes:
            return False

        if func_decl.is_overloaded_operator() or func_decl.is_operator:
            return False

        return True

    def _collect_symbol_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []

        for templ in self._function_templates:
            if not self._is_eligible_template(templ):
                continue
            name = templ.function.name
            if name not in seen:
                seen.add(name)
                names.append(name)

        return names

    def _render(
        self,
        *,
        with_imports: bool,
        with_shim_stream: bool,
    ):
        self.Imports.add(
            "from numbast.static.function_template import bind_static_function_templates"
        )

        for symbol in self._collect_symbol_names():
            if symbol not in self._function_symbols:
                self._function_symbols.append(symbol)

        encoded_templates = base64.b64encode(
            pickle.dumps(
                self._function_templates, protocol=pickle.HIGHEST_PROTOCOL
            )
        ).decode("ascii")

        rendered_bindings = self.function_templates_binding_template.format(
            encoded_function_templates=encoded_templates,
            skip_prefix=repr(self._skip_prefix),
            skip_non_device=repr(self._skip_non_device),
            exclude=repr(self._excludes),
            arg_intent=repr(self._function_argument_intents),
        )

        output = ""
        if with_imports:
            output += "\n" + get_rendered_imports()

        if with_shim_stream:
            output += "\n" + get_shim('""')
            output += "\n" + get_callconv_utils()

        output += "\n" + rendered_bindings
        self._output = output

    def render_as_str(
        self,
        *,
        with_imports: bool,
        with_shim_stream: bool,
    ) -> str:
        self._render(
            with_imports=with_imports,
            with_shim_stream=with_shim_stream,
        )
        return self._output
