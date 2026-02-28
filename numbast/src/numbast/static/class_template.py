# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import pickle

from ast_canopy.decl import ClassTemplate

from numbast.class_template import (
    ConcreteTypeCache as _class_template_cache,
    bind_cxx_class_templates,
)
from numbast.static.renderer import (
    BaseRenderer,
    get_callconv_utils,
    get_rendered_imports,
    get_shim,
)


def clear_class_template_cache():
    """Clear the concrete type cache used by dynamic class-template binders."""
    _class_template_cache.clear()


def _decode_pickled_class_templates(
    encoded_class_templates: str,
) -> list[ClassTemplate]:
    if not encoded_class_templates:
        return []
    raw = base64.b64decode(encoded_class_templates.encode("ascii"))
    return pickle.loads(raw)


def bind_static_class_templates(
    *,
    namespace: dict[str, object],
    encoded_class_templates: str,
    shim_writer,
    header_path: str,
    excludes: list[str] | None = None,
    arg_intent: dict | None = None,
) -> list[object]:
    """
    Materialize and bind class templates into a generated static module namespace.

    Parameters
    ----------
    namespace : dict[str, object]
        Usually ``globals()`` of the generated binding module.
    encoded_class_templates : str
        Base64-encoded pickle payload containing ``list[ClassTemplate]``.
    shim_writer : object
        Shim writer adapter exposed by the generated static binding script.
    header_path : str
        Header path used by class-template specialization parsing.
    excludes : list[str] | None
        Class template names (short or qualified) to exclude from binding.
    arg_intent : dict | None
        Optional argument intent overrides.

    Returns
    -------
    list[object]
        Bound class-template handles.
    """
    class_templates = _decode_pickled_class_templates(encoded_class_templates)
    exclude_names = set(excludes or [])

    filtered_templates = [
        ct
        for ct in class_templates
        if ct.record.name not in exclude_names
        and ct.record.qual_name not in exclude_names
    ]

    apis = bind_cxx_class_templates(
        class_templates=filtered_templates,
        header_path=header_path,
        shim_writer=shim_writer,
        arg_intent=arg_intent,
    )
    for api in apis:
        namespace[api.__name__] = api

    return apis


class StaticClassTemplatesRenderer(BaseRenderer):
    """Render static bindings for class templates."""

    class_templates_binding_template = """
bind_static_class_templates(
    namespace=globals(),
    encoded_class_templates=\"\"\"{encoded_class_templates}\"\"\",
    shim_writer=shim_writer,
    header_path={header_path},
    excludes={excludes},
    arg_intent={arg_intent},
)
"""

    def __init__(
        self,
        class_templates: list[ClassTemplate],
        *,
        header_path: str,
        excludes: list[str] | None = None,
        function_argument_intents: dict | None = None,
    ):
        super().__init__(class_templates)
        self._class_templates = class_templates
        self._header_path = header_path
        self._excludes = excludes or []
        self._function_argument_intents = function_argument_intents or {}

    def _collect_symbol_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        exclude_names = set(self._excludes)

        for templ in self._class_templates:
            if (
                templ.record.name in exclude_names
                or templ.record.qual_name in exclude_names
            ):
                continue

            name = templ.record.qual_name
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
            "from numbast.static.class_template import bind_static_class_templates"
        )

        for symbol in self._collect_symbol_names():
            if symbol not in self._record_symbols:
                self._record_symbols.append(symbol)

        encoded_templates = base64.b64encode(
            pickle.dumps(
                self._class_templates, protocol=pickle.HIGHEST_PROTOCOL
            )
        ).decode("ascii")

        rendered_bindings = self.class_templates_binding_template.format(
            encoded_class_templates=encoded_templates,
            header_path=repr(self._header_path),
            excludes=repr(self._excludes),
            arg_intent=repr(self._function_argument_intents),
        )

        output = ""
        if with_imports:
            output += "\n" + get_rendered_imports()

        if with_shim_stream:
            output += "\n" + get_shim(f'"#include <{self._header_path}>"')
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
