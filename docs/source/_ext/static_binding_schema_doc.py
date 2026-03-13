# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any

import yaml


def load_yaml_schema(schema_path: Path) -> dict[str, Any]:
    with schema_path.open(encoding="utf-8") as schema_file:
        schema = yaml.safe_load(schema_file)

    if not isinstance(schema, dict):
        raise ValueError(
            f"Schema at {schema_path} must deserialize to a mapping object."
        )
    return schema


def _format_type(spec: dict[str, Any]) -> str:
    schema_type = spec.get("type")
    if isinstance(schema_type, list):
        return " | ".join(str(part) for part in schema_type)
    if isinstance(schema_type, str):
        return schema_type

    if "oneOf" in spec:
        one_of = [
            _format_type(item)
            for item in spec["oneOf"]
            if isinstance(item, dict)
        ]
        return "oneOf(" + ", ".join(one_of) + ")"

    if "properties" in spec:
        return "object"

    return "any"


def _format_default(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, int, float)):
        return str(value)
    dumped = yaml.safe_dump(value, sort_keys=False).strip()
    return dumped.replace("\n", " ")


def _collect_constraints(spec: dict[str, Any]) -> list[str]:
    constraints: list[str] = []

    if "enum" in spec:
        allowed = ", ".join(f"``{value}``" for value in spec["enum"])
        constraints.append(f"Allowed values: {allowed}")
    if "pattern" in spec:
        constraints.append(f"Pattern: ``{spec['pattern']}``")
    if "minItems" in spec:
        constraints.append(f"Min items: {spec['minItems']}")
    if "maxItems" in spec:
        constraints.append(f"Max items: {spec['maxItems']}")

    items = spec.get("items")
    if isinstance(items, dict):
        constraints.append(f"Item type: ``{_format_type(items)}``")

    if spec.get("additionalProperties") is False:
        constraints.append("No unspecified sub-keys")

    return constraints


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).split())


def _append_wrapped(
    lines: list[str],
    text: str,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> None:
    wrapped = textwrap.fill(
        text,
        width=120,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    lines.extend(wrapped.splitlines())


def _render_properties_deflist(
    lines: list[str],
    properties: dict[str, Any],
    indent: str = "",
) -> None:
    body_indent = indent + "   "
    for key, spec in properties.items():
        if not isinstance(spec, dict):
            continue

        type_label = _format_type(spec)
        description = _normalize_text(
            spec.get("description", "No description.")
        )

        lines.append(f"{indent}``{key}`` : ``{type_label}``")

        _append_wrapped(
            lines,
            description,
            initial_indent=body_indent,
            subsequent_indent=body_indent,
        )

        if "default" in spec:
            default_value = _format_default(spec["default"])
            lines.append("")
            lines.append(f"{body_indent}Default: ``{default_value}``.")

        constraints = _collect_constraints(spec)
        if constraints:
            lines.append("")
            lines.append(f"{body_indent}Constraints:")
            lines.append("")
            for constraint in constraints:
                lines.append(f"{body_indent}- {constraint}")

        examples = spec.get("examples")
        if isinstance(examples, list) and examples:
            lines.append("")
            lines.append(f"{body_indent}Example:")
            lines.append("")
            lines.append(f"{body_indent}.. code-block:: yaml")
            lines.append("")
            for example in examples:
                dumped = yaml.safe_dump(
                    {key: example}, default_flow_style=False, sort_keys=False
                ).rstrip()
                for dump_line in dumped.splitlines():
                    lines.append(f"{body_indent}   {dump_line}")
            lines.append("")

        lines.append("")


def render_static_binding_schema_reference(
    schema: dict[str, Any],
    schema_text: str,
    schema_repo_path: str,
) -> str:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Schema 'properties' must be a mapping.")

    required_keys = schema.get("required", [])
    if not isinstance(required_keys, list):
        raise ValueError("Schema 'required' must be a list.")
    required_set = set(required_keys)

    required_props = {k: v for k, v in properties.items() if k in required_set}
    optional_props = {
        k: v for k, v in properties.items() if k not in required_set
    }

    lines: list[str] = [
        "This section is generated directly from:",
        f"``{schema_repo_path}``",
        "",
        "Required keys",
        "--------------",
        "",
    ]

    _render_properties_deflist(lines, required_props)

    lines.extend(
        [
            "Optional keys",
            "--------------",
            "",
        ]
    )

    _render_properties_deflist(lines, optional_props)

    nested = [
        (key, spec)
        for key, spec in optional_props.items()
        if isinstance(spec, dict) and isinstance(spec.get("properties"), dict)
    ]
    if nested:
        lines.extend(
            [
                "Optional nested keys",
                "^^^^^^^^^^^^^^^^^^^^",
                "",
            ]
        )
        for key, spec in nested:
            lines.append(f".. rubric:: ``{key}``")
            lines.append("")
            child_properties = spec.get("properties", {})
            _render_properties_deflist(lines, child_properties)

    lines.extend(
        [
            "Raw schema",
            "----------",
            "",
            ".. code-block:: yaml",
            "",
        ]
    )

    for raw_line in schema_text.rstrip().splitlines():
        if not raw_line.strip():
            lines.append("")
            continue

        leading_spaces = len(raw_line) - len(raw_line.lstrip(" "))
        content = raw_line.lstrip(" ")
        wrapped_lines = textwrap.wrap(
            content,
            width=max(20, 117 - leading_spaces),
            break_long_words=False,
            break_on_hyphens=False,
        )
        for wrapped_line in wrapped_lines:
            lines.append(f"   {' ' * leading_spaces}{wrapped_line}")

    lines.append("")
    return "\n".join(lines)


def generate_static_binding_schema_reference(
    schema_path: Path,
    output_path: Path,
    schema_repo_path: str,
) -> None:
    schema = load_yaml_schema(schema_path)
    schema_text = schema_path.read_text(encoding="utf-8")
    rendered = render_static_binding_schema_reference(
        schema=schema,
        schema_text=schema_text,
        schema_repo_path=schema_repo_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
